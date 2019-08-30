import time
import numpy as np
import pandas as pd
import itertools
from pathlib import Path

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import ChannelList
from qcodes.dataset.measurements import Measurement
from qcodes.utils.validators import Strings, Numbers, Ints, Lists

from shockley import get_data_from_ds
from shockley.drivers.devices.generic import COND_QUANT, LCC_MAP, DIRECT_MAP, \
                                             _dfdx, binomial, _meas_gate_leak, \
                                             devJSONEncoder, parse_json_dump,\
                                             Result, Contact, Gate, DummyChan
from shockley.drivers.devices.fet import SingleFET
from shockley.sweeps import do1d

from shockplot import start_listener, listener_is_running
from shockplot.qcodes_dataset import QCSubscriber


class HallBar(Instrument):

    def __init__(self, name, ohmics=None, gate=None,
                 chip_carrier=None, dev_store='./_dev_store', **kwargs):

        # make sure the instrument and measurement attributes
        # exist but are None unless added explicitly
        self._mdac = None
        self._current = None
        self._volt = None
        self._vxx = None
        self._vxy = None

        if chip_carrier is None:
            print('using direct pin mapping')
            self.pin_map = DIRECT_MAP
        elif chip_carrier=='lcc':
            print('using lcc pin mapping')
            self.pin_map = LCC_MAP
        else:
            raise ValueError('pin to MDAC mapping not defined')

        self.dev_store = Path(dev_store).resolve()
        self.dev_store.mkdir(parents=True, exist_ok=True)

        super().__init__(name, **kwargs)

        ### Ohmics/Gate setup ###
        if gate is None:
            raise ValueError('define a gate contact')
        else:
            g = DummyChan(self, f'{self.name}_gate', gate)
            self.add_submodule('gate', g)
            self.add_parameter('gate_pin',
                            label='Gate Pin',
                            set_cmd=None,
                            initial_value=gate)

        ### create dummy ohmic submodule(s)
        if ohmics is None:
            raise ValueError('define ohmic contacts')
        else:
            self.add_parameter('ohmic_pins',
                            label='Ohmic Pins',
                            set_cmd=None,
                            initial_value=ohmics,
                            validators=Lists(Ints()))

            ohms = ChannelList(self, "Ohmics", DummyChan,
                                   snapshotable=True)
            for i, o in enumerate(self.ohmic_pins()):
                ohm = DummyChan(self, f'{self.name}_ohmic{i}', o)
                ohms.append(ohm)
            ohms.lock()
            self.add_submodule('ohmics', ohms)

        ### add FET submodules ###
        pairs = list(itertools.combinations(self.ohmics,2))
        self.segments = [None]*len(pairs)
        for i, pair in enumerate(pairs):
            s = pair[1].chip_number()
            d = pair[0].chip_number()

            s_name = f'{name}_segment{i:02d}'
            try:
                inst = Instrument.find_instrument(s_name)
                inst.close()
                # print('\t...closed')
            except Exception as e:
                # print(f'exception when closing: {e}')
                pass

            self.segments[i] = SingleFET(s_name, source=s, drain=d, gate=self.gate_pin(), chip_carrier=chip_carrier)
            self.add_submodule(f'segment{i:02d}', self.segments[i])

    def add_instruments_2probe(self, mdac=None, smu_curr=None, smu_volt=None):
        ''' add instruments to make measurements. this is optional if you are
            only loading the device for analysis. '''

        if (mdac is None) or (smu_curr is None):
            print('[WARNING]: instruments not loaded. provide at MDAC and SMU_CURR. (SMU_VOLT optional).')
            return None

        self._mdac = mdac
        self._current = smu_curr
        self._volt = smu_volt
        if self._volt is not None:
            self._volt.step = 0.005
            self._volt.inter_delay = 0.01

        # overwrite dummy gate submodule
        g = Gate(self, f'{self.name}_gate', self.gate_pin())
        del self.submodules['gate']
        self.add_submodule('gate', g)
        self.gate.voltage.step = 0.005
        self.gate.voltage.inter_delay = 0.01
        self.gate._set_limits(minlimit=-5.0, maxlimit=5.0, ratelimit=5.0)
        self.gate_step_coarse = 0.02 # V
        self.gate_step_fine = 0.001 # V
        self.gate_delay = 0.05

        # overwrite dummy source submodules
        ohms = ChannelList(self, "Ohmics", Contact,
                               snapshotable=True)
        for i, o in enumerate(self.ohmic_pins()):
            ohm = Contact(self, f'{self.name}_ohmic{i}', o)
            ohms.append(ohm)
        ohms.lock()
        del self.submodules['ohmics']
        self.add_submodule('ohmics', ohms)

        for o in self.ohmics:
            o.voltage.step = 0.005
            o.voltage.inter_delay = 0.01
            o._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)

        for i, (seg, pair) in enumerate(zip(self.segments, itertools.combinations(self.ohmics,2))):
            s = pair[1]
            d = pair[0]

            seg._mdac = self._mdac
            seg._current = self._current
            seg._volt = self._volt

            del seg.submodules['drain']
            seg.add_submodule('drain', d)

            del seg.submodules['source']
            seg.add_submodule('source', s)

            del seg.submodules['gate']
            seg.add_submodule('gate', self.gate)

            seg.connect = lambda: self._connect_segment(i)
            seg.disconnect = lambda: self._disconnect_segment(i)
            seg.meas_gate_leak = self.meas_gate_leak

            try:
                seg.load()
            except FileNotFoundError:
                pass

    def add_instruments_4probe(xx_volt=None, xy_volt=None,
                               magx=None, magy=None, magz=None):

        # load lockin parameters for 4-probe hall measurements
        if (xx_volt is None) or (xy_volt is None):
            print('[WARNING]: instruments not loaded. provide at MDAC and SMU_CURR. (SMU_VOLT optional).')
            return None

        self._vxx = xx_volt
        self._vxy = xy_volt


    def close(self):
        # overwrite default close() so that this closes all of the individual wires as well
        for seg in self.segments:
            if seg is not None:
                seg.close()

        super().close()

    def _connect_segment(self, i):

        self.gate.microd_to_dac() # should already be connected

        self.segments[i].drain.microd_to_bus()

        self.segments[i].source.voltage(0.0)
        self.segments[i].source.microd_to_dac()

        # float the other sources
        s = self.segments[i].source.chip_number()
        d = self.segments[i].drain.chip_number()
        for o in self.ohmics:
            if o.chip_number() not in [s, d]:
                o.microd_float()

    def _disconnect_segment(self, i):
        self.segments[i].source.voltage(0.0)
        self.segments[i].source.microd_float()
        self.segments[i].drain.microd_float()

    def disconnect(self):

        for o in self.ohmics:
            o.voltage(0.0)
            o.terminate()

    def as_dataframe(self):
        # concatenate all the DataFrames from each segment
        frames = []
        for seg in self.segments:
            frames.append(seg.as_dataframe())

        df = pd.concat(frames, ignore_index=True)
        del df['connect']
        return df

    def meas_gate_leak(self, overwrite = False):

        seg0 = self.segments[0]
        if (overwrite==False) and (seg0.gate_max.val() is not None):
            run_id = self.segments[0].gate_max.run_id()
            print(f'Gate leakage for {self.name} already measured in run {run_id}.')
            return run_id

        if not listener_is_running():
            start_listener()

        if self._volt:
            volt_set = self._volt
            for o in self.ohmics:
                o.terminate()
            self.gate.microd_to_bus()
        else:
            volt_set = self.gate.voltage
            self.ohmics[-1].microd_to_bus()
            for o in self.ohmics[:-1]:
                o.microd_float()
            self.gate.microd_to_dac()

        run_id, vmax = _meas_gate_leak(volt_set, self._current, self.gate_step_coarse,
                                       5.0, self.gate_delay,
                                       di_limit=seg0.leakage_threshold(), compliance=2e-9)

        for seg in self.segments:
            # record results
            seg.gate_leak_runid.append(run_id)
            seg.gate_min.val(min(-1*vmax, -1*self.gate_step_coarse))
            seg.gate_min.run_id(run_id)
            seg.gate_max.val(max(vmax, self.gate_step_coarse))
            seg.gate_max.run_id(run_id)

            # adjust limits
            seg.V_open(min(seg.gate_max.val(), seg.V_open()))
            if seg.V_open() < 0.1:
                seg.V_open(0.0)

            seg.gate._set_limits(minlimit=seg.gate_min.val(), maxlimit=seg.gate_max.val())
            if vmax < 0.5:
                seg.gate.failed(True)

            seg.save()

        return run_id

    def label_bad_contacts(self):

        df = pd.DataFrame(index = range(len(self.segments)), columns = ['drain', 'source', 'R', 'passed'])

        for i, seg in enumerate(self.segments):
            s = seg.source.chip_number()
            d = seg.drain.chip_number()
            passed = (seg.R_open.val() < 2e6) and (seg.R_open.val() > 0)
            df.loc[i] = (d, s, seg.R_open.val(), passed)

        olist = [o.chip_number() for o in self.ohmics]
        working = [True]*len(self.ohmics)
        for i in range(len(self.ohmics)):
            o = self.ohmics[i]
            cn = olist[i]
            passed = (df[(df['source']==cn) | (df['drain']==cn)]['passed'].values)
            if np.any(passed):
                working[i] = True
                o.failed(False)
            else:
                working[i] = False
                o.failed(True)

        if binomial(sum(working),2)!=sum(df['passed']):
            print('[WARNING] Results of contact test inconsistent. Hall bar likely discontinuous.')

        return [(cn,'good') if w else (cn,'bad') for cn, w in zip(olist,working)]
