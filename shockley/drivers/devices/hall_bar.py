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


class HallBar_2P(Instrument):
    ''' 2 probe checks for Hall bar at zero field '''

    def __init__(self, name, md, smu_curr, ohmics=None, gate=None,
                 smu_volt=None, chip_carrier=None, **kwargs):
        '''
        This assumes I have an SMU connected to the bus of the MDAC
        and the cryostat is directly connected to the MDAC microd connections.

        Args:
            name (str): the name of the device
            md (MDAC): MDAC that the device is connected to
            smu_curr (qcodes.Parameter): qcodes parameter to measure current
            smu_volt (qcodes.Parameter): qcodes parameter to set voltage on SMU
            source (int): socket number of source contact
            drain (int): socket number of drain contact
            gate (int): socket number of gate
            chip_carrier (str): type of chip carrier used
                                (allows for mapping between socket numbers and MDAC numbers)

            **kwargs are passed to qcodes.instrument.base.Instrument
        '''

        self._mdac = md
        self._current = smu_curr
        self._volt = smu_volt
        if self._volt is not None:
            self._volt.step = 0.005
            self._volt.inter_delay = 0.01

        self.COND_QUANT =  7.748091729e-5 # Siemens

        if chip_carrier is None:
            self._pin_map = DIRECT_MAP
        elif chip_carrier=='lcc':
            self._pin_map = LCC_MAP
        else:
            raise ValueError('pin to MDAC mapping not defined')

        super().__init__(name, **kwargs)

        ### check some inputs ###
        if ohmics is None:
            # this is only a kwarg for readability
            raise ValueError('define list of ohmic contacts')
        else:
            self.contacts_list = ohmics

        if gate is None:
            # this is only a kwarg for readability
            raise ValueError('define a gate contact')

        ### create ohmic submodule(s)
        contacts = ChannelList(self, "Conacts", Contact,
                               snapshotable=True)

        for c in self.contacts_list:
            contact = Contact(self,'{}_c{:02}'.format(name, c), c)
            contacts.append(contact)
            self.add_submodule('c{:02}'.format(c), contact)
        contacts.lock()
        self.add_submodule('contacts', contacts)

        ### create gate submodule
        g = Gate(self, f'{name}_gate', gate)
        self.add_submodule('gate', g)

        self.failed = False
        self.contacts_failed = [False for o in self.contacts_list]
        self.v_bias = 1e-3 # V
        self.V_open = 3.0 # V
        self.gate_leak_thresh = 400e-12 # A
        self.r_limit = 2e6 # enough to distingush from leakage current @ 1mV bias
        self.hysteresis_swing = [0.5, 1.0, 1.5]

        self.gate.voltage.step = 0.005
        self.gate.voltage.inter_delay = 0.01
        self.gate._set_limits(minlimit=-5.0, maxlimit=5.0, ratelimit=5.0)
        self.gate_min = None
        self.gate_max = None
        self.gate_step_coarse = 0.02 # V
        self.gate_step_fine = 0.001 # V
        self.gate_delay = 0.05

        for ohm in self.contacts:
            ohm.voltage.step = 0.005
            ohm.voltage.inter_delay = 0.01
            ohm._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)

        self.gate_leak_runid = []
        self.pinchoff_runid = []
        self.ss_runid = []
        self.hysteresis_runid = []

    def disconnect(self):

        self.contacts.voltage(0.0)
        self.contacts.microd_float()

    def meas_gate_leak(self, compliance=2e-9, plot_logs=False, write_period=0.1):

        if not listener_is_running():
            start_listener()

        volt_limit = self.gate._get_maxlimit()
        volt_step = self.gate_step_coarse
        curr_limit = self.gate_leak_thresh

        meas = Measurement()
        meas.write_period = write_period

        if self._volt:
            volt_set = self._volt
        else:
            volt_set = self.gate.voltage

        meas.register_parameter(volt_set)
        volt_set.post_delay = 0

        meas.register_parameter(self._current, setpoints=(volt_set,))

        with meas.run() as ds:

            plot_subscriber = QCSubscriber(ds.dataset, volt_set, [self._current],
                                           grid=None, log=plot_logs)
            ds.dataset.subscribe(plot_subscriber)

            curr = np.array([])
            for vg in np.arange(0, volt_limit+volt_step, volt_step):

                volt_set.set(vg)
                time.sleep(self.gate_delay)

                curr = np.append(curr, self._current.get())
                ds.add_result((volt_set, vg),
                              (self._current, curr[-1]))

                if vg>0:
                    if np.abs(curr.max() - curr.min()) > curr_limit:
                        print('Leakage current limit exceeded!')
                        vmax = vg-volt_step # previous step was the limit
                        break
                    elif np.abs(curr[-1])>compliance:
                        print('Current compliance level exceeded!')
                        vmax = vg-volt_step # previous step was the limit
                        break
            else:
                vmax = volt_limit

            for vg in np.arange(vmax, 0, -1*volt_step):

                volt_set.set(vg)
                time.sleep(self.gate_delay)

                curr = np.append(curr, self._current.get())
                ds.add_result((volt_set, vg),
                              (self._current, curr[-1]))

            self.gate_min = min(-1*vmax, -1*self.gate_step_coarse)
            self.gate_max = max(vmax, self.gate_step_coarse)
            self.gate._set_limits(minlimit=self.gate_min, maxlimit=self.gate_max)
            self.V_open = min(self.gate_max, self.V_open)
            self.gate_leak_runid.append(ds.run_id)

            if vmax < 0.5:
                self.failed  = True
            else:
                self.failed = False

        return self.gate_min, self.gate_max

    def meas_connectivity(self):
        ''' test pairs of contacts to see what is connected '''

        self.gate.microd_to_dac()
        self.gate.voltage(self.V_open)
        self.contacts.voltage(0.0)
        self.contacts.microd_float()

        mi = pd.MultiIndex.from_tuples(
                list(itertools.combinations(self.contacts_list,2)))
        df = pd.DataFrame(index=mi, columns=['runid', 'resistance'])

        xarray = np.linspace(-0.025, 0.025, 251)
        for pair in itertools.combinations(self.contacts,2):

            s = pair[1].chip_number()
            d = pair[0].chip_number()
            print(f'{s}->{d}')

            pair[0].microd_to_bus()
            pair[1].microd_to_dac()

            runid = do1d(pair[1].voltage, xarray, 0.05, self._current)

            dd = get_data_from_ds(runid, self._current.name, dtype='numpy')

            bias = dd['x']['vals']
            current = dd['y']['vals']
            popt = np.polyfit(bias, current, 1)
            R = 1/popt[0]
            df.loc[d,s] = [runid,R]

            pair[1].voltage(0.0)
            pair[1].microd_float()
            pair[0].microd_float()

        idx = df[(df['resistance']<self.r_limit) & (df['resistance']>0)].index.values
        working = list(set(c for row in idx for c in row)) # list of working contacts

        # update list of failed contacts
        self.contacts_failed = [False if c in working else True for c in self.contacts_list]
