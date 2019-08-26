import time
import numpy as np
import pandas as pd
import itertools

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import ChannelList
from qcodes.dataset.measurements import Measurement

from shockley import get_data_from_ds
from shockley.drivers.devices.generic import _dfdx, Contact, Gate, \
                                             LCC_MAP, DIRECT_MAP
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
            self.pin_map = DIRECT_MAP
        elif chip_carrier=='lcc':
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
        pairs = itertools.combinations(self.ohmics,2)
        self.segments = [None]*len(pairs)
        for i, pair in enumerate(pairs):
            s = pair[1].chip_number()
            d = pair[0].chip_number()

            s_name = f'{name}_segment{i}'
            try:
                inst = Instrument.find_instrument(s_name)
                inst.close()
                # print('\t...closed')
            except Exception as e:
                # print(f'exception when closing: {e}')
                pass

            self.segments[i] = SingleFET(s_name, source=s, drain=d, gate=self.gate_pin(), chip_carrier=chip_carrier)
            self.add_submodule(f'segment{i}', self.segments[i])

    def add_instruments_dc(self, mdac=None, smu_curr=None, smu_volt=None, xx_volt=None, xy_volt=None):
        ''' add instruments to make measurements. this is optional if you are
            only loading the device for analysis. '''

        if (mdac is None) or (smu_curr is None):
            print('[WARNING]: instruments not loaded. provide at least MDAC and SMU_CURR.')
            return None

        self._mdac = None
        self._current = None
        self._volt = None

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
            s = pair[1].chip_number()
            d = pair[0].chip_number()
            seg._mdac = self._mdac
            seg._current = self._current; seg._volt = self._volt
            seg.drain = d; seg.source = s
            seg.gate = self.gate
            seg.connect = lambda: self._connect_segment(i)
            seg.disconnect = lambda: self._disconnect_segment(i)
            seg.meas_gate_leak = self.meas_gate_leak

            try:
                seg.load()
            except FileNotFoundError:
                pass

        # load lockin parameters for 4-probe hall measurements
        if xx_volt is not None:
            self._vxx = xx_volt
        if xy_volt is not None:
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
        for o on self.ohmics:
            if o.chip_number not in [s, d]:
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

        df = pd.concat(frames)
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
                                       self.gate._get_maxlimit(), self.gate_delay,
                                       di_limit=seg0.leakage_threshold(), compliance=0.5e-9)

        for seg in self.segments:
            # record results
            seg.gate_leak_runid.append(run_id)
            seg.gate_min.val(min(-1*vmax, -1*self.gate_step_coarse))
            seg.gate_min.run_id(run_id)
            seg.gate_max.val(max(vmax, self.gate_step_coarse))
            seg.gate_max.run_id(run_id)

            # adjust limits
            seg.V_open(min(seg.gate_max.val(), seg.V_open()))
            seg.gate._set_limits(minlimit=seg.gate_min.val(), maxlimit=seg.gate_max.val())

            if vmax < 0.5:
                seg.gate.failed(True)

            seg.save()

        return run_id
