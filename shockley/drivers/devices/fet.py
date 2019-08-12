''' trying to come up with a reasonable way to address a device
    connected through an MDAC.

    currently just works for a single MDAC '''

from pathlib import Path
import time
import json
import numpy as np
from scipy.optimize import curve_fit
from qcodes.instrument.base import Instrument
from qcodes.dataset.measurements import Measurement

from shockley import get_data_from_ds
from shockley.drivers.parameters import CounterParam
from shockley.drivers.devices.generic import _dfdx, Contact, Gate, \
                                             LCC_MAP, DIRECT_MAP, devJSONEncoder
from shockley.sweeps import do1d, do1d_repeat_twoway, gen_sweep_array

from shockplot import start_listener, listener_is_running
from shockplot.qcodes_dataset import QCSubscriber

LCC_MAP = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8,
    9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
    17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24,
    25:26, 26:27, 27:28, 28:29, 29:30, 30:31, 31:32, 32:33,
}

DIRECT_MAP = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8,
    9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
    17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24,
    25:25, 26:26, 27:27, 28:28, 29:29, 30:30, 31:31, 32:32,
    33:33, 34:34, 35:35, 36:36, 37:37, 38:38, 39:39, 40:40,
    41:41, 42:42, 43:43, 44:44, 45:45, 46:46, 47:47, 48:48,
}

def cond_stats(v_gate, cond):
    # calculate maximum conductance and transconductance
    # sort by gate voltage
    inds    = v_gate.argsort()
    v_gate  = v_gate[inds]
    cond    = cond[inds]

    dGdV    = _dfdx(cond, v_gate)

    try:
        # extract threshold and mobility by fitting data between 10 and 80 percent of max current.
        ind10 = [ n for n, i in enumerate(cond-0.1*np.nanmax(cond)) if i < 0 ][-1]
        ind80 = [ n for n, i in enumerate(cond-0.8*np.nanmax(cond)) if i < 0 ][-1]

        G_max = cond[ind80:].max()
        dGdV_max = dGdV[ind10:ind80].max()
        return G_max, dGdV_max
    except:
        print('\tWARNING: could not determine pinchoff range for G_max calculation. Fall back to full array.')
        return cond.max(), dGdV.max()

def _sub_threshold_func(gate_voltage, a, b, c):
    # Base 10 exponential fit to extract volt/decade
    return a*10**(b*gate_voltage)+c

def sub_threshold_fit(v_gate, I):
        # Function to fit sub-threshold swing. It is assumed that dataset is a full pinch-off curve.

        # Flip data left-right dependent on sweep direction
        inds    = v_gate.argsort()
        v_gate  = v_gate[inds]
        I       = I[inds]

        # Get threshold voltage
        Vth = trans_cond_fit(v_gate, I)[0][0]

        # Only fit data below threshold voltage
        ind_vth = np.argmin(np.abs(v_gate-Vth))
        v_gate  = v_gate[0:ind_vth]
        I       = I[0:ind_vth]

        # Initial guess parameters
        p_guess = [I[-1],                           # Current at threshold
                1/(v_gate[-1]-v_gate[np.argmin(abs(I-I[-1]/10))]),    # Estimated decay rate
                0]

        # Do fit on log scale to ensure reasonably sized numbers
        popt, pcov = curve_fit(_sub_threshold_func,v_gate-Vth,I,p0=p_guess)
        perr = np.sqrt(np.diag(pcov))

        # Fix translation of x-axis in fitting
        popt[0] = popt[0]*10**(-popt[1]*Vth)

        # Calculate goodness of fit
        ss_res = np.sum((I - _sub_threshold_func(v_gate,*popt)) ** 2)
        ss_tot = np.sum((I - np.mean(I)) ** 2)
        R2 = 1 - (ss_res / ss_tot)

        return popt, perr, R2

def _trans_func(gate_voltage, Vth, Rs, Gm):

    # Analytic pinch-off curve
    I = (gate_voltage - Vth)/(Rs*(gate_voltage - Vth) + 1/Gm)
    I *= (gate_voltage >= Vth) # Set I = 0 for gate_voltage < Vth
    return I

def trans_cond_fit(v_gate, I):

    # sort by gate voltage
    inds    = v_gate.argsort()
    v_gate  = v_gate[inds]
    I       = I[inds]

    # extract threshold and mobility by fitting data between 10 and 80 percent of max current.
    ind10 = [ n for n, i in enumerate(I-0.1*np.nanmax(I)) if i < 0 ][-1]
    ind80 = [ n for n, i in enumerate(I-0.8*np.nanmax(I)) if i < 0 ][-1]

    # Scale numbers to around unity for better fitting
    I_scaled = I/np.nanmax(I)
    # Initial guess parameters
    p_guess = [v_gate[ind10], 1/I_scaled[-1],
                (I_scaled[ind80] - I_scaled[ind10])/(v_gate[ind80]-v_gate[ind10])]

    popt, pcov = curve_fit(_trans_func,v_gate[ind10:ind80],I_scaled[ind10:ind80],
                                p0=p_guess,bounds=([-10, 0, 0], [10, np.infty, np.infty]))
    perr = np.sqrt(np.diag(pcov))

    # Fix scaling
    popt[1] /= np.nanmax(I)
    popt[2] *= np.nanmax(I)
    perr[1] /= np.nanmax(I)
    perr[2] *= np.nanmax(I)

    # Calculate goodness of fit
    ss_res = np.sum((I[ind10:ind80] - _trans_func(v_gate[ind10:ind80],*popt)) ** 2)
    ss_tot = np.sum((I[ind10:ind80] - np.mean(I[ind10:ind80])) ** 2)
    R2 = 1 - (ss_res / ss_tot)

    return popt, perr, R2

def v_th_stats(x_data, y_data, window_length=10, y_threshold = 0.05, x_error = 1e-3):

    ## Eoin original
    sort_keys = np.argsort(x_data)
    x_data = x_data[sort_keys]
    y_data = y_data[sort_keys]

    # clean data
    y_data[y_data < y_threshold/5.0] = 0.0

    # compute the local variance using window_length
    window = np.ones((window_length,))/window_length
    l_sq = np.convolve(y_data**2,window)[:-len(window)+1]
    l_mu = np.convolve(y_data,window)[:-len(window)+1]
    l_var = l_sq - l_mu**2

    l_var_streak  = np.zeros(l_var.shape) # streak counter

    # start from most negative gate voltage
    # look for device to turn 'on'
    # keep track of minimum local variance as gate voltage increases (min_l_var)
    for i, lv in enumerate(l_var[::-1]):
        if i==0:
            min_l_var = lv
        else:
            if lv != 0.:
                min_l_var = min(min_l_var, lv)

            # if local variance is less than 50*min_l_var and
            # the y value is less than the threshold
            # count it as part of a streak of 'pinched off' values
            if (lv < 50 * min_l_var) and (y_data[-i-1] < y_threshold):
                l_var_streak[-i-1] = l_var_streak[-i] + 1

    zeros = np.where(l_var_streak==0)
    Vth_index = zeros[0][0] # get the index marking the end of the first streak

    if (x_data[Vth_index]-x_data[0]) > x_error:
        x_off = x_data[Vth_index]
    else:
        # too close to the end
        x_off = None

    return x_off #, l_var, l_var_streak

class FET(Instrument):
    ''' FET device class '''

    def __init__(self, name, md, smu_curr, source=None, drain=None, gate=None,
                 smu_volt=None, chip_carrier=None,  pickle_path='./_pickle_jar', **kwargs):
        '''
        This assumes I have an SMU connected to the bus of the MDAC
        and the cryostat is directly connected to the MDAC microd connections.

        Args:
            name (str): the name of the device
            md (MDAC): MDAC that the device is connected to
            curr_param (qcodes.Parameter): qcodes parameter to measure current
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

        self.pickle_path = Path(pickle_path).resolve()
        self.pickle_path.mkdir(parents=True, exist_ok=True)

        super().__init__(name, **kwargs)

        ### check some inputs ###
        if source is None:
            # this is only a kwarg for readability
            raise ValueError('define a source contact')

        if drain is None:
            # this is only a kwarg for readability
            raise ValueError('define a drain contact')

        if gate is None:
            # this is only a kwarg for readability
            raise ValueError('define a gate contact')

        ### create source submodule
        s = Contact(self, f'{name}_source', source)
        self.add_submodule('source', s)

        ### create gate submodule
        g = Gate(self, f'{name}_gate', gate)
        self.add_submodule('gate', g)

        ### create drain submodule
        d = Contact(self, f'{name}_drain', drain)
        self.add_submodule('drain', d)

        self.failed = False

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

        self.source.voltage.step = 0.005
        self.source.voltage.inter_delay = 0.01
        self.source._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)
        self.source_bias = 1e-3 # V

        self.drain.voltage.step = 0.005
        self.drain.voltage.inter_delay = 0.01
        self.drain._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)

        self.gate_leak_runid = []
        self.pinchoff_runid = []
        self.ss_runid = []
        self.hysteresis_runid = []

    def connect(self, bias=0.0, gate=0.0):

        self.drain.microd_to_bus()
        self.source.microd_to_dac()
        self.gate.microd_to_dac()

    def disconnect(self):

        self.drain.voltage(0.0)
        self.source.voltage(0.0)

        self.drain.microd_float()
        self.source.microd_float()

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
            self.source.terminate()
            self.drain.terminate()
            self.gate.microd_to_bus()
        else:
            volt_set = self.gate.voltage
            self.source.float()
            self.drain.microd_to_bus()
            self.gate.microd_to_dac()

        meas.register_parameter(volt_set)
        volt_set.post_delay = 0

        meas.register_parameter(self._current, setpoints=(volt_set,))

        with meas.run() as ds:

            plot_subscriber = QCSubscriber(ds.dataset, volt_set, [self._current],
                                           grid=None, log=plot_logs)
            ds.dataset.subscribe(plot_subscriber)

            curr = np.array([])
            for vg in gen_sweep_array(0, volt_limit+volt_step, step=volt_step):

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

            for vg in gen_sweep_array(vmax, 0, step=volt_step):

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

        self.drain.microd_to_bus()
        self.source.microd_to_dac()

        xarray = gen_sweep_array(-0.025, 0.025, num=251)
        runid = do1d(self.source.voltage, xarray, 0.05, self._current)

        dd = get_data_from_ds(runid, self._current.name, dtype='numpy')

        bias = dd['x']['vals']
        current = dd['y']['vals']
        popt = np.polyfit(bias, current, 1)
        R = 1/popt[0]
        self.r_open = R

        if (R > self.r_limit) or (R < 0):
            self.failed = True

    def meas_pinch_off(self, exit_on_vth = False,
                       send_grid=True, plot_logs=False, write_period=0.1):

        if not listener_is_running():
            start_listener()

        meas = Measurement()
        meas.register_parameter(self.gate.voltage)
        self.gate.voltage.post_delay = 0
        meas.register_parameter(self._current, setpoints=(self.gate.voltage,))

        self.source.voltage(self.source_bias)
        self.gate.voltage(self.V_open)

        xarray = gen_sweep_array(self.V_open, self.gate_min-self.gate_step_coarse, step=self.gate_step_coarse)
        current = np.full(len(xarray), np.nan, dtype=np.float)

        meas.write_period = write_period

        with meas.run() as ds:

            if send_grid:
                grid = [xarray]
            else:
                grid = None

            plot_subscriber = QCSubscriber(ds.dataset, self.gate.voltage, [self._current],
                                           grid=grid, log=plot_logs)
            ds.dataset.subscribe(plot_subscriber)

            for i,x in enumerate(xarray):

                self.gate.voltage.set(x)
                time.sleep(self.gate_delay)

                current[i] = self._current.get()
                ds.add_result((self.gate.voltage, x),
                              (self._current, current[i]))

                if exit_on_vth and i>20:
                    # exit if a threshold voltage is found that is
                    # at least 0.5V from the most negative gate voltage
                    vth = v_th_stats(xarray[0:i], current[0:i]/self.source_bias/self.COND_QUANT,
                                     window_length=20, y_threshold=0.005, x_error = 0.5)
                    if vth is not None:
                        self.failed=False
                        print('Threshold voltage found. Exiting sweep.')
                        break

            time.sleep(write_period) # let final data points propogate to plot
            self.pinchoff_runid.append(ds.run_id)

    def meas_sub_threshold(self,  send_grid=True, plot_logs=False, write_period=0.1):

        if hasattr(self, 'V_th_fit')==False or (self.V_th_fit is None):
            print('ERROR: cannot measure subthreshold region without a value for V_th_fit.')
            return 0

        if not listener_is_running():
            start_listener()

        self.source.voltage(self.source_bias)
        self.gate.voltage(self.V_open)

        v1 = self.V_open
        v2 = self.V_th_fit + 0.5
        v3 = max(self.V_th_fit - 1.0, self.gate_min)
        xarray = np.concatenate((gen_sweep_array(v1, v2, step=self.gate_step_coarse),
                                 gen_sweep_array(v2-self.gate_step_fine, v3, step=self.gate_step_fine)))

        run_id = do1d(self.gate.voltage, xarray, self.gate_delay, self._current)

        self.ss_runid.append(run_id)

    def meas_hysteresis(self, delayy = 2.0):

        if hasattr(self, 'V_th_stat')==False or (self.V_th_stat is None):
            print('ERROR: cannot measure hysteresis without a value for V_th_stat.')
            return 0

        runids = [None, None, None]
        for i, swing in enumerate(self.hysteresis_swing):

            vmax = self.V_open
            vmin  = self.V_th_stat - swing

            if self.gate_min > vmin:
                continue
            else:
                self.source.voltage(self.source_bias)
                self.gate.voltage(self.V_open)

                xarray = gen_sweep_array(self.V_open, vmin-self.gate_step_coarse,
                                         step=self.gate_step_coarse)

                runids[i] = do1d_repeat_twoway(self.gate.voltage, xarray, self.gate_delay,
                                               2, delayy, self._current)

        self.hysteresis_runid.append(runids)

    def fit_pinchoff(self, runid):
        # fit to trans conductance curve
        df = get_data_from_ds(runid, self._current.name, dtype='pandas')
        v_gate = df.index.values.flatten()
        i_sd = df.values.flatten()

        try:
            popt, perr, R2 = trans_cond_fit(v_gate,i_sd)

            # store data in device object
            self.Gm = popt[2]*1e6 # uS
            self.Gm_std = perr[2]*1e6 # uS
            self.V_th_fit = popt[0] # V
            self.V_th_fit_std = perr[0] # V
            self.I_sat = popt[1] # A
            self.I_sat_std = perr[1] # A
            self.trans_cond_R2 = R2
        except IndexError as e:
            print(f'\tWARNING: could not determine pinchoff range for trans conductance fit. Giving up.')

    def get_pinchoff_stats(self, runid):

        df = get_data_from_ds(runid, self._current.name, dtype='pandas')
        v_gate = df.index.values.flatten()
        g_sd = df.values.flatten()/self.source_bias/self.COND_QUANT

        self.G_max, self.dGdV_max = cond_stats(v_gate, g_sd)
        self.V_th_stat = v_th_stats(v_gate, g_sd, window_length=20, y_threshold=0.01)

    def fit_sub_threshold(self, runid):
        # fit to trans conductance curve
        df = get_data_from_ds(runid, self._current.name, dtype='pandas')
        v_gate = df.index.values.flatten()
        i_sd = df.values.flatten()

        try:
            popt, perr, R2 = sub_threshold_fit(v_gate,i_sd)

            # store data in device object
            self.sub_threshold_swing = 1/popt[1] # V/decade
            self.sub_threshold_swing_R2 = R2
        except Exception as e:
            print(f'\tWARNING: could fit subthreshold swing. MSG: {e}')

    def get_hysteresis(self, runids):

        for swing, runid in zip(self.hysteresis_swing, runids):
            dd = get_data_from_ds(runid, self._current.name, dtype='numpy')
            v_gate = dd['x']['vals']
            g_sd = dd['z']['vals']/self.source_bias/self.COND_QUANT

            vth_down = v_th_stats(v_gate, g_sd[0], window_length=20, y_threshold=0.01)
            vth_up = v_th_stats(v_gate, g_sd[1], window_length=20, y_threshold=0.01)

            setattr(self, 'hysteresis_' + str(int(10*swing)).zfill(3), vth_down-vth_up)

    def save(self):
        from pprint import pprint
        jstr = json.dumps(self.__dict__, cls = devJSONEncoder)
        pprint( jstr )
