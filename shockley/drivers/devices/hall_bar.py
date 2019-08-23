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


class HallBarDC(Instrument):

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
                            label='Source Pin',
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
    #     self.segments = [None]*len(self.source_pins())
    #     for i, s in enumerate(self.source_pins()):
    #         s_name = f'{name}_segment{i}'
    #         try:
    #             inst = Instrument.find_instrument(s_name)
    #             inst.close()
    #             print('\t...closed')
    #         except Exception as e:
    #             # print(f'exception when closing: {e}')
    #             pass
    #
    #         self.segments[i] = SingleFET(s_name, source=s, drain=self.drain_pin(), gate=self.gate_pin(), chip_carrier=chip_carrier)
    #         self.add_submodule(f'segment{i}', self.segments[i])
    #
    # def add_instruments_dc(self, mdac=None, smu_curr=None, smu_volt=None, xx_volt=None, xy_volt=None):
    #     ''' add instruments to make measurements. this is optional if you are
    #         only loading the device for analysis. '''
    #
    #     if (mdac is None) or (smu_curr is None):
    #         print('[WARNING]: instruments not loaded. provide MDAC and SMU_CURR.')
    #         return None
    #
    #     self._mdac = mdac
    #     self._current = smu_curr
    #     self._volt = smu_volt
    #     if self._volt is not None:
    #         self._volt.step = 0.005
    #         self._volt.inter_delay = 0.01
    #
    #     # overwrite dummy gate submodule
    #     g = Gate(self, f'{self.name}_gate', self.gate_pin())
    #     del self.submodules['gate']
    #     self.add_submodule('gate', g)
    #     self.gate.voltage.step = 0.005
    #     self.gate.voltage.inter_delay = 0.01
    #     self.gate._set_limits(minlimit=-5.0, maxlimit=5.0, ratelimit=5.0)
    #     self.gate_step_coarse = 0.02 # V
    #     self.gate_step_fine = 0.001 # V
    #     self.gate_delay = 0.05
    #
    #     # overwrite dummy drain submodule
    #     d = Contact(self, f'{self.name}_drain', self.drain_pin())
    #     del self.submodules['drain']
    #     self.add_submodule('drain', d)
    #     self.drain.voltage.step = 0.005
    #     self.drain.voltage.inter_delay = 0.01
    #     self.drain._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)
    #
    #     # overwrite dummy source submodules
    #     srcs = ChannelList(self, "Sources", Contact,
    #                            snapshotable=True)
    #     for i, s in enumerate(self.source_pins()):
    #         src = Contact(self, f'{self.name}_segment{i}_source', s)
    #         srcs.append(src)
    #     srcs.lock()
    #     del self.submodules['sources']
    #     self.add_submodule('sources', srcs)
    #     for src in self.sources:
    #         src.voltage.step = 0.005
    #         src.voltage.inter_delay = 0.01
    #         src._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)
    #
    #     for i, (seg, src) in enumerate(zip(self.segments, self.sources)):
    #         seg._mdac = self._mdac
    #         seg._current = self._current; seg._volt = self._volt
    #         seg.drain = self.drain; seg.gate = self.gate
    #         seg.source = src
    #         seg.connect = lambda: self._connect_segment(i)
    #         seg.disconnect = lambda: self._disconnect_segment(i)
    #         seg.meas_gate_leak = self.meas_gate_leak
    #
    #         try:
    #             seg.load()
    #         except FileNotFoundError:
    #             pass
    #
    # def close(self):
    #     # overwrite default close() so that this closes all of the individual wires as well
    #     for seg in self.segments:
    #         if seg is not None:
    #             seg.close()
    #
    #     super().close()
    #
    # def _connect_segment(self, i):
    #
    #     # should already be connected
    #     self.gate.microd_to_dac()
    #     self.drain.microd_to_bus()
    #
    #     self.sources[j].voltage(0.0)
    #     self.sources[i].microd_to_dac()
    #
    #     # float the other sources
    #     for j in [n for n in range(0,4) if n!=i]:
    #         self.sources[j].voltage(0.0)
    #         self.sources[j].microd_float()
    #
    #
    # def _disconnect_segment(self, i):
    #     self.sources[i].voltage(0.0)
    #     self.sources[i].microd_float()
    #
    # def disconnect(self):
    #
    #     self.drain.terminate()
    #
    #     for j in range(0,4):
    #         self.sources[j].voltage(0.0)
    #         self.sources[j].terminate()
    #
    # def as_dataframe(self):
    #     # concatenate all the DataFrames from each segment
    #     frames = []
    #     for seg in self.segments:
    #         frames.append(seg.as_dataframe())
    #
    #     df = pd.concat(frames)
    #     del df['connect']
    #     return df
    #
    # def meas_gate_leak(self, overwrite = False):
    #
    #     seg0 = self.segments[0]
    #     if (overwrite==False) and (seg0.gate_max.val() is not None):
    #         run_id = self.segments[0].gate_max.run_id()
    #         print(f'Gate leakage for {self.name} already measured in run {run_id}.')
    #         return run_id
    #
    #     if not listener_is_running():
    #         start_listener()
    #
    #     if self._volt:
    #         volt_set = self._volt
    #         self.drain.terminate()
    #         for src in self.sources:
    #             src.terminate()
    #         self.gate.microd_to_bus()
    #     else:
    #         volt_set = self.gate.voltage
    #         self.drain.microd_to_bus()
    #         for src in self.sources:
    #             src.microd_float()
    #         self.gate.microd_to_dac()
    #
    #     run_id, vmax = _meas_gate_leak(volt_set, self._current, self.gate_step_coarse,
    #                                    self.gate._get_maxlimit(), self.gate_delay,
    #                                    di_limit=seg0.leakage_threshold(), compliance=0.5e-9)
    #
    #
    #     for seg in self.segments:
    #         # record results
    #         seg.gate_leak_runid.append(run_id)
    #         seg.gate_min.val(min(-1*vmax, -1*self.gate_step_coarse))
    #         seg.gate_min.run_id(run_id)
    #         seg.gate_max.val(max(vmax, self.gate_step_coarse))
    #         seg.gate_max.run_id(run_id)
    #
    #         # adjust limits
    #         seg.V_open(min(seg.gate_max.val(), seg.V_open()))
    #         seg.gate._set_limits(minlimit=seg.gate_min.val(), maxlimit=seg.gate_max.val())
    #
    #         if vmax < 0.5:
    #             seg.gate.failed(True)
    #
    #         seg.save()
    #
    #     return run_id

    # ''' 2 probe checks for Hall bar at zero field '''
    #
    # def __init__(self, name, md, smu_curr, ohmics=None, gate=None,
    #              smu_volt=None, chip_carrier=None, **kwargs):
    #     '''
    #     This assumes I have an SMU connected to the bus of the MDAC
    #     and the cryostat is directly connected to the MDAC microd connections.
    #
    #     Args:
    #         name (str): the name of the device
    #         md (MDAC): MDAC that the device is connected to
    #         smu_curr (qcodes.Parameter): qcodes parameter to measure current
    #         smu_volt (qcodes.Parameter): qcodes parameter to set voltage on SMU
    #         source (int): socket number of source contact
    #         drain (int): socket number of drain contact
    #         gate (int): socket number of gate
    #         chip_carrier (str): type of chip carrier used
    #                             (allows for mapping between socket numbers and MDAC numbers)
    #
    #         **kwargs are passed to qcodes.instrument.base.Instrument
    #     '''
    #
    #     self._mdac = md
    #     self._current = smu_curr
    #     self._volt = smu_volt
    #     if self._volt is not None:
    #         self._volt.step = 0.005
    #         self._volt.inter_delay = 0.01
    #
    #     self.COND_QUANT =  7.748091729e-5 # Siemens
    #
    #     if chip_carrier is None:
    #         self._pin_map = DIRECT_MAP
    #     elif chip_carrier=='lcc':
    #         self._pin_map = LCC_MAP
    #     else:
    #         raise ValueError('pin to MDAC mapping not defined')
    #
    #     super().__init__(name, **kwargs)
    #
    #     ### check some inputs ###
    #     if ohmics is None:
    #         # this is only a kwarg for readability
    #         raise ValueError('define list of ohmic contacts')
    #     else:
    #         self.contacts_list = ohmics
    #
    #     if gate is None:
    #         # this is only a kwarg for readability
    #         raise ValueError('define a gate contact')
    #
    #     ### create ohmic submodule(s)
    #     contacts = ChannelList(self, "Conacts", Contact,
    #                            snapshotable=True)
    #
    #     for c in self.contacts_list:
    #         contact = Contact(self,'{}_c{:02}'.format(name, c), c)
    #         contacts.append(contact)
    #         self.add_submodule('c{:02}'.format(c), contact)
    #     contacts.lock()
    #     self.add_submodule('contacts', contacts)
    #
    #     ### create gate submodule
    #     g = Gate(self, f'{name}_gate', gate)
    #     self.add_submodule('gate', g)
    #
    #     self.failed = False
    #     self.contacts_failed = [False for o in self.contacts_list]
    #     self.v_bias = 1e-3 # V
    #     self.V_open = 3.0 # V
    #     self.gate_leak_thresh = 400e-12 # A
    #     self.r_limit = 2e6 # enough to distingush from leakage current @ 1mV bias
    #     self.hysteresis_swing = [0.5, 1.0, 1.5]
    #
    #     self.gate.voltage.step = 0.005
    #     self.gate.voltage.inter_delay = 0.01
    #     self.gate._set_limits(minlimit=-5.0, maxlimit=5.0, ratelimit=5.0)
    #     self.gate_min = None
    #     self.gate_max = None
    #     self.gate_step_coarse = 0.02 # V
    #     self.gate_step_fine = 0.001 # V
    #     self.gate_delay = 0.05
    #
    #     for ohm in self.contacts:
    #         ohm.voltage.step = 0.005
    #         ohm.voltage.inter_delay = 0.01
    #         ohm._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)
    #
    #     self.gate_leak_runid = []
    #     self.pinchoff_runid = []
    #     self.ss_runid = []
    #     self.hysteresis_runid = []
    #
    # def disconnect(self):
    #
    #     self.contacts.voltage(0.0)
    #     self.contacts.microd_float()
    #
    # def meas_gate_leak(self, compliance=2e-9, plot_logs=False, write_period=0.1):
    #
    #     if not listener_is_running():
    #         start_listener()
    #
    #     volt_limit = self.gate._get_maxlimit()
    #     volt_step = self.gate_step_coarse
    #     curr_limit = self.gate_leak_thresh
    #
    #     meas = Measurement()
    #     meas.write_period = write_period
    #
    #     if self._volt:
    #         volt_set = self._volt
    #     else:
    #         volt_set = self.gate.voltage
    #
    #     meas.register_parameter(volt_set)
    #     volt_set.post_delay = 0
    #
    #     meas.register_parameter(self._current, setpoints=(volt_set,))
    #
    #     with meas.run() as ds:
    #
    #         plot_subscriber = QCSubscriber(ds.dataset, volt_set, [self._current],
    #                                        grid=None, log=plot_logs)
    #         ds.dataset.subscribe(plot_subscriber)
    #
    #         curr = np.array([])
    #         for vg in np.arange(0, volt_limit+volt_step, volt_step):
    #
    #             volt_set.set(vg)
    #             time.sleep(self.gate_delay)
    #
    #             curr = np.append(curr, self._current.get())
    #             ds.add_result((volt_set, vg),
    #                           (self._current, curr[-1]))
    #
    #             if vg>0:
    #                 if np.abs(curr.max() - curr.min()) > curr_limit:
    #                     print('Leakage current limit exceeded!')
    #                     vmax = vg-volt_step # previous step was the limit
    #                     break
    #                 elif np.abs(curr[-1])>compliance:
    #                     print('Current compliance level exceeded!')
    #                     vmax = vg-volt_step # previous step was the limit
    #                     break
    #         else:
    #             vmax = volt_limit
    #
    #         for vg in np.arange(vmax, 0, -1*volt_step):
    #
    #             volt_set.set(vg)
    #             time.sleep(self.gate_delay)
    #
    #             curr = np.append(curr, self._current.get())
    #             ds.add_result((volt_set, vg),
    #                           (self._current, curr[-1]))
    #
    #         self.gate_min = min(-1*vmax, -1*self.gate_step_coarse)
    #         self.gate_max = max(vmax, self.gate_step_coarse)
    #         self.gate._set_limits(minlimit=self.gate_min, maxlimit=self.gate_max)
    #         self.V_open = min(self.gate_max, self.V_open)
    #         self.gate_leak_runid.append(ds.run_id)
    #
    #         if vmax < 0.5:
    #             self.failed  = True
    #         else:
    #             self.failed = False
    #
    #     return self.gate_min, self.gate_max
    #
    # def meas_connectivity(self):
    #     ''' test pairs of contacts to see what is connected '''
    #
    #     self.gate.microd_to_dac()
    #     self.gate.voltage(self.V_open)
    #     self.contacts.voltage(0.0)
    #     self.contacts.microd_float()
    #
    #     mi = pd.MultiIndex.from_tuples(
    #             list(itertools.combinations(self.contacts_list,2)))
    #     df = pd.DataFrame(index=mi, columns=['runid', 'resistance'])
    #
    #     xarray = np.linspace(-0.025, 0.025, 251)
    #     for pair in itertools.combinations(self.contacts,2):
    #
    #         s = pair[1].chip_number()
    #         d = pair[0].chip_number()
    #         print(f'{s}->{d}')
    #
    #         pair[0].microd_to_bus()
    #         pair[1].microd_to_dac()
    #
    #         runid = do1d(pair[1].voltage, xarray, 0.05, self._current)
    #
    #         dd = get_data_from_ds(runid, self._current.name, dtype='numpy')
    #
    #         bias = dd['x']['vals']
    #         current = dd['y']['vals']
    #         popt = np.polyfit(bias, current, 1)
    #         R = 1/popt[0]
    #         df.loc[d,s] = [runid,R]
    #
    #         pair[1].voltage(0.0)
    #         pair[1].microd_float()
    #         pair[0].microd_float()
    #
    #     idx = df[(df['resistance']<self.r_limit) & (df['resistance']>0)].index.values
    #     working = list(set(c for row in idx for c in row)) # list of working contacts
    #
    #     # update list of failed contacts
    #     self.contacts_failed = [False if c in working else True for c in self.contacts_list]
