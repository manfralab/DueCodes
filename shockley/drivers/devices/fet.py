''' trying to come up with a reasonable way to address a device
    connected through an MDAC.

    currently just works for a single MDAC '''

from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.dataset.measurements import Measurement
from qcodes.utils.validators import Strings, Numbers, Ints, Lists

from shockley import get_data_from_ds
from shockley.drivers.parameters import CounterParam
from shockley.drivers.devices.generic import COND_QUANT, LCC_MAP, DIRECT_MAP, \
                                             _dfdx, _meas_gate_leak, \
                                             devJSONEncoder, parse_json_dump,\
                                             Result, Contact, Gate, DummyChan



from shockley.sweeps import do1d, do1d_repeat_twoway, gen_sweep_array

from shockplot import start_listener, listener_is_running
from shockplot.qcodes_dataset import QCSubscriber

### analysis ###

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

### device class ###

class SingleFET(Instrument):
    ### TO DO:
    # subclass this into a 4 wire device with shared drain... somehow

    ''' FET device class '''

    def __init__(self, name, source=None, drain=None, gate=None,
                 chip_carrier=None,  dev_store='./_dev_store', **kwargs):
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

        # make sure the instrument and measurement attributes
        # exist but are None unless added explicitly
        self._mdac = None
        self._current = None
        self._volt = None

        if chip_carrier is None:
            self.pin_map = DIRECT_MAP
        elif chip_carrier=='lcc':
            self.pin_map = LCC_MAP
        else:
            raise ValueError('pin to MDAC mapping not defined')

        self.dev_store = Path(dev_store).resolve()
        self.dev_store.mkdir(parents=True, exist_ok=True)

        super().__init__(name, **kwargs)

        ### Source/Drain/Gate setup ###
        if source is None:
            raise ValueError('define a source contact')
        else:
            s = DummyChan(self, f'source', source)
            self.add_submodule('source', s)
            self.add_parameter('source_pin',
                            label='Source Pin',
                            set_cmd=None,
                            initial_value=source)
        if drain is None:
            raise ValueError('define a drain contact')
        else:
            d = DummyChan(self, f'drain', drain)
            self.add_submodule('drain', d)
            self.add_parameter('drain_pin',
                            set_cmd=None,
                            label='Drain Pin',
                            initial_value=drain)
        if gate is None:
            raise ValueError('define a gate contact')
        else:
            g = DummyChan(self, f'gate', gate)
            self.add_submodule('gate', g)
            self.add_parameter('gate_pin',
                            label='Gate Pin',
                            set_cmd=None,
                            initial_value=gate)

        # device information
        self.add_parameter('length',
                        label='Design Length of FET channel',
                        set_cmd=None,
                        get_cmd=None,
                        unit='um',
                        vals=Numbers())
        self.add_parameter('width',
                        label='Design Width of SAG structure',
                        unit='nm',
                        set_cmd=None,
                        get_cmd=None,
                        vals=Numbers())
        self.add_parameter('structure',
                        label='SAG Structure',
                        set_cmd=None,
                        get_cmd=None,
                        vals=Strings())

        # track history
        self.gate_leak_runid = []
        self.iv_runid = []
        self.pinchoff_runid = []
        self.ss_runid = []
        self.hysteresis_runid = []

        # parameters to define measurements
        self.add_parameter('leakage_threshold',
                        label='Gate Leakage Threshold',
                        unit='A',
                        set_cmd=None,
                        initial_value=400e-12)
        self.add_parameter('gate_threshold',
                        label='Cutoff For Useful Gates',
                        unit='V',
                        set_cmd=None,
                        initial_value=0.5)
        self.add_parameter('resistance_threshold',
                        label='Resistance Threshold',
                        unit='Ohm',
                        set_cmd=None,
                        initial_value=2e6)
        self.add_parameter('hysteresis_Vswing',
                        set_cmd=None,
                        label='Voltage Swings for Hysterises Measurments',
                        unit='V',
                        initial_value=(0.5,1,1.5))
        self.add_parameter('V_open',
                        label='Gate Open Voltage',
                        unit='V',
                        set_cmd=None,
                        initial_value=3)
        self.add_parameter('V_bias',
                        label='Bias Voltage',
                        unit='V',
                        set_cmd=None,
                        initial_value=0.001)

        # create results submodule(s)
        results_names = ['gate_min', 'gate_max', 'R_open',
                        'Gm', 'Gm_std', 'V_th_fit', 'V_th_fit_std',
                        'I_sat', 'I_sat_std', 'trans_cond_R2',
                        'G_max', 'dGdV_max', 'V_th_stat',
                        'sub_threshold_swing', 'sub_threshold_swing_R2' ] + \
                        ['hysteresis_'+str(int(10*sw)).zfill(3) for sw in self.hysteresis_Vswing()]

        results_units = ['V', 'V', 'Ohm',
                        'uS', 'uS', 'V', 'V',
                        '2e^2/h', '2e^2/h/V', '',
                        'A', 'A', 'V',
                        'V/decade', ''] + \
                        ['V']*len(self.hysteresis_Vswing())

        for res_name, res_unit in zip(results_names, results_units):
            res = Result(self, res_name, res_unit)
            self.add_submodule(res_name, res)

        try:
            self.load()
        except Exception as e:
            print(f'[ERROR] (while loading saved parameters): {e}')


    def add_instruments(self, mdac=None, smu_curr=None, smu_volt=None):
        ''' add instruments to make measurements. this is optional if you are
            only loading the device for analysis. '''

        if (mdac is None) or (smu_curr is None):
            print('[WARNING]: instruments not loaded. provide MDAC and SMU_CURR.')
            return None

        self._mdac = mdac
        self._current = smu_curr
        self._volt = smu_volt
        if self._volt is not None:
            self._volt.step = 0.005
            self._volt.inter_delay = 0.01

        # overwrite dummy source submodule
        s = Contact(self, f'source', self.source_pin())
        del self.submodules['source']
        self.add_submodule('source', s)
        self.source.voltage.step = 0.005
        self.source.voltage.inter_delay = 0.01
        self.source._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)

        # overwrite dummy gate submodule
        g = Gate(self, f'gate', self.gate_pin())
        del self.submodules['gate']
        self.add_submodule('gate', g)
        self.gate.voltage.step = 0.005
        self.gate.voltage.inter_delay = 0.01
        self.gate._set_limits(minlimit=-5.0, maxlimit=5.0, ratelimit=5.0)
        self.gate_step_coarse = 0.02 # V
        self.gate_step_fine = 0.001 # V
        self.gate_delay = 0.05

        # overwrite dummy drain submodule
        d = Contact(self, f'drain', self.drain_pin())
        del self.submodules['drain']
        self.add_submodule('drain', d)
        self.drain.voltage.step = 0.005
        self.drain.voltage.inter_delay = 0.01
        self.drain._set_limits(minlimit=-0.5, maxlimit=0.5, ratelimit=5.0)

        try:
            self.load()
        except Exception as e:
            print(f'[ERROR] (while loading saved parameters): {e}')

    def connect(self):

        self.drain.voltage(0.0)
        self.drain.microd_to_bus()

        self.source.voltage(0.0)
        self.source.microd_to_dac()

        self.gate.microd_to_dac()

    def disconnect(self):

        if self.gate.failed():
            self.gate.voltage(0.0)
            self.gate.terminate()

            self.drain.voltage(0.0)
            self.drain.microd_float()

            self.source.voltage(0.0)
            self.source.microd_float()
        else:
            self.drain.voltage(0.0)
            self.drain.terminate()

            self.source.voltage(0.0)
            self.source.microd_float()

    def working(self):
        if any([self.source.failed(), self.drain.failed(), self.gate.failed()]):
            return False
        else:
            return True

    def connect_leakage(self):

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

    def meas_gate_leak(self, overwrite = False):

        if (overwrite==False) and (self.gate_max.val() is not None):
            run_id = self.gate_max.run_id()
            print(f'Gate leakage for {self.name} already measured in run {run_id}.')
            return run_id

        if not listener_is_running():
            start_listener()

        run_id, vmax = _meas_gate_leak(volt_set, self._current, self.gate_step_coarse,
                                       self.gate._get_maxlimit(), self.gate_delay,
                                       di_limit=self.leakage_threshold(), compliance=2e-9)

        # record results
        self.gate_leak_runid.append(run_id)
        self.gate_min.val(min(-1*vmax, -1*self.gate_step_coarse))
        self.gate_min.run_id(run_id)
        self.gate_max.val(max(vmax, self.gate_step_coarse))
        self.gate_max.run_id(run_id)

        # adjust limits
        self.V_open(min(self.gate_max.val(), self.V_open()))
        self.gate._set_limits(minlimit=self.gate_min.val(), maxlimit=self.gate_max.val())

        if vmax < 0.5:
            self.gate.failed(True)

        self.save()

        return run_id

    def meas_connectivity(self, overwrite = False):
        ''' test pairs of contacts to see what is connected '''

        if (overwrite==False) and (self.R_open.val() is not None):
            run_id = self.R_open.run_id()
            print(f'Connectivity for {self.name} already measured in run {run_id}.')
            return run_id

        self.gate.voltage(self.V_open())

        xarray = gen_sweep_array(-0.025, 0.025, num=251)
        run_id = do1d(self.source.voltage, xarray, 0.05, self._current)

        dd = get_data_from_ds(run_id, self._current.name, dtype='numpy')

        bias = dd['x']['vals']
        current = dd['y']['vals']
        popt = np.polyfit(bias, current, 1)
        R = 1/popt[0]

        self.R_open.val(R)
        self.R_open.run_id(run_id)
        self.iv_runid.append(run_id)

        if (R > self.resistance_threshold()) or (R < 0):
            self.source.failed(True)
            self.drain.failed(True)

        self.save()

        return run_id

    def meas_pinch_off(self, exit_on_vth = False, overwrite = False,
                       send_grid=True, plot_logs=False, write_period=0.1):

        if (overwrite==False) and (len(self.pinchoff_runid)>0):
            print(f'Pinchoff for {self.name} already measured in run(s) {self.pinchoff_runid}. '
                   'Returning most recent run.')
            return self.pinchoff_runid[-1]

        if not listener_is_running():
            start_listener()

        meas = Measurement()
        meas.register_parameter(self.gate.voltage)
        self.gate.voltage.post_delay = 0
        meas.register_parameter(self._current, setpoints=(self.gate.voltage,))

        self.source.voltage(self.V_bias())
        self.gate.voltage(self.V_open())
        time.sleep(0.5)

        xarray = gen_sweep_array(self.V_open(), self.gate_min.val()-self.gate_step_coarse,
                                 step=self.gate_step_coarse)
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

                self.gate.voltage(x)
                time.sleep(self.gate_delay)

                current[i] = self._current()
                ds.add_result((self.gate.voltage, x),
                              (self._current, current[i]))

                if exit_on_vth and i>20:
                    # exit if a threshold voltage is found that is
                    # at least 0.5V from the most negative gate voltage
                    vth = v_th_stats(xarray[0:i], current[0:i]/self.V_bias()/COND_QUANT,
                                     window_length=20, y_threshold=0.005, x_error = 0.5)
                    if vth is not None:
                        print('Threshold voltage found. Exiting sweep.')
                        break

            time.sleep(write_period) # let final data points propogate to plot
            self.pinchoff_runid.append(ds.run_id)

            self.save()

            return ds.run_id

    def meas_sub_threshold(self, overwrite = False,
                           send_grid=True, plot_logs=False, write_period=0.1):

        if (self.V_th_fit.val() is None):
            print('ERROR: cannot measure subthreshold region without a value for V_th_fit.')
            return -1

        if (overwrite==False) and (len(self.ss_runid)):
            print(f'Gate leakage for {self.name} already measured in run(s) {self.ss_runid}.'
                    ' Returning most recent run.')
            return self.ss_runid[-1]

        if not listener_is_running():
            start_listener()

        self.source.voltage(self.V_bias())
        self.gate.voltage(self.V_open())
        time.sleep(0.5)

        v1 = self.V_open()
        v2 = self.V_th_fit.val() + 0.25
        v3 = max(self.V_th_fit.val() - 1.25, self.gate_min.val())
        xarray = np.concatenate((gen_sweep_array(v1, v2, step=self.gate_step_coarse),
                                 gen_sweep_array(v2-self.gate_step_fine, v3, step=self.gate_step_fine)))

        run_id = do1d(self.gate.voltage, xarray, self.gate_delay, self._current)

        self.ss_runid.append(run_id)

        self.save()
        return run_id

    def meas_hysteresis(self, delayy = 2.0, overwrite=False):

        if (self.V_th_stat.val() is None):
            print('ERROR: cannot measure hysteresis without a value for V_th_stat.')
            return -1

        if (overwrite==False) and (len(self.hysteresis_runid)>0):
            print(f'Pinchoff for {self.name} already measured in runs {self.hysteresis_runid}. '
                   'Returning most recent runs.')
            return self.hysteresis_runid[-1]

        runids = [None, None, None]
        for i, swing in enumerate(self.hysteresis_Vswing()):

            vmax = self.V_open()
            vmin  = self.V_th_stat.val() - swing

            if self.gate_min.val() > vmin:
                continue
            else:
                self.source.voltage(self.V_bias())
                self.gate.voltage(self.V_open())

                xarray = gen_sweep_array(self.V_open(), vmin-self.gate_step_coarse,
                                         step=self.gate_step_coarse)

                runids[i] = do1d_repeat_twoway(self.gate.voltage, xarray, self.gate_delay,
                                               2, delayy, self._current)

        self.hysteresis_runid.append(runids)
        self.save()
        return runids

    def fit_pinchoff(self, runid):
        # fit to trans conductance curve
        df = get_data_from_ds(runid, self._current.name, dtype='pandas')
        v_gate = df.index.values.flatten()
        i_sd = df.values.flatten()

        try:
            popt, perr, R2 = trans_cond_fit(v_gate,i_sd)

            # store data in device object
            self.Gm.val(popt[2]*1e6), self.Gm.run_id(runid);
            self.Gm_std.val(perr[2]*1e6), self.Gm_std.run_id(runid);
            self.V_th_fit.val(popt[0]), self.V_th_fit.run_id(runid);
            self.V_th_fit_std.val(perr[0]), self.V_th_fit_std.run_id(runid);
            self.I_sat.val(popt[1]), self.I_sat.run_id(runid);
            self.I_sat_std.val(perr[1]), self.I_sat_std.run_id(runid);
            self.trans_cond_R2.val(R2), self.trans_cond_R2.run_id(runid);

            xfit = v_gate; yfit = _trans_func(v_gate, *popt);
        except IndexError as e:
            print(f'\tWARNING: could not determine pinchoff range for trans conductance fit. Giving up.')
            xfit = np.array([]); yfit = np.array([]);

        self.save()
        return xfit, yfit

    def get_pinchoff_stats(self, runid):

        df = get_data_from_ds(runid, self._current.name, dtype='pandas')
        v_gate = df.index.values.flatten()
        g_sd = df.values.flatten()/self.V_bias()/COND_QUANT

        G_max, dGdV_max = cond_stats(v_gate, g_sd)
        V_th_stat = v_th_stats(v_gate, g_sd, window_length=20, y_threshold=0.01)

        self.G_max.val(G_max), self.G_max.run_id(runid);
        self.dGdV_max.val(dGdV_max), self.dGdV_max.run_id(runid);
        self.V_th_stat.val(V_th_stat), self.V_th_stat.run_id(runid);

        self.save()

        return V_th_stat, G_max, dGdV_max

    def fit_sub_threshold(self, runid):

        # fit to trans conductance curve
        df = get_data_from_ds(runid, self._current.name, dtype='pandas')
        v_gate = df.index.values.flatten()
        i_sd = df.values.flatten()

        idx = np.argsort(v_gate)
        v_gate = v_gate[idx]
        i_sd = i_sd[idx]

        try:
            popt, perr, R2 = sub_threshold_fit(v_gate,i_sd)
            ind_vth = np.argmin(np.abs(v_gate-self.V_th_fit.val()))

            # store data in device object
            self.sub_threshold_swing.val(1/popt[1]), self.sub_threshold_swing.run_id(runid);
            self.sub_threshold_swing_R2.val(R2), self.sub_threshold_swing_R2.run_id(runid);
            xfit = v_gate; yfit = _sub_threshold_func(v_gate, *popt)

            xfit  = v_gate[0:ind_vth]
            yfit  = _sub_threshold_func(xfit, *popt)
        except Exception as e:
            print(f'\tWARNING: could fit subthreshold swing. MSG: {e}')
            xfit = np.array([]); yfit = np.array([]);

        self.save()
        return xfit, yfit

    def get_hysteresis(self, runids):

        v_th_results = []
        for swing, runid in zip(self.hysteresis_Vswing(), runids):
            dd = get_data_from_ds(runid, self._current.name, dtype='numpy')
            v_gate = dd['x']['vals']
            g_sd = dd['z']['vals']/self.V_bias()/COND_QUANT

            vth_down = v_th_stats(v_gate, g_sd[0], window_length=20, y_threshold=0.01)
            vth_up = v_th_stats(v_gate, g_sd[1], window_length=20, y_threshold=0.01)

            try:
                hyst = getattr(self, 'hysteresis_' + str(int(10*swing)).zfill(3))
                hyst.val(vth_down-vth_up); hyst.run_id(runid);
            except TypeError:
                # handles the case of one of the thresholds being None
                pass
            v_th_results.append((vth_down, vth_up))

        self.save()
        return v_th_results

    def _to_json(self):

        jstr = json.dumps(self.__dict__, cls = devJSONEncoder, indent=4, sort_keys=True)
        return jstr

    def save(self):

        jstr = self._to_json()

        fpath = self.dev_store / Path(f'{self.name}.json')
        with fpath.open('w+') as f:
            f.write(jstr)

    def load(self):

        fpath = self.dev_store / Path(f'{self.name}.json')
        with fpath.open('r') as f:
            jstr = f.read()

        if jstr != '':

            jdict = parse_json_dump(jstr, device=self)

            if 'pin_map' in jdict:
                self.pin_map = {int(k):v for k,v in self.pin_map.items()}

            # convert pickle_path to proper type
            if 'dev_store' in jdict:
                path = jdict['dev_store']
                if path is not None:
                    self.dev_store = Path(jdict['dev_store']).resolve()

            # update a few limits/tags
            if self.gate_max.val() is not None:
                self.V_open(min(self.gate_max.val(), self.V_open()))
                if self.gate_max.val() < 0.5:
                    self.gate.failed(True)

                if isinstance(self.gate, Gate):
                    self.gate._set_limits(maxlimit=self.gate_max.val())
                    self.gate._set_limits(minlimit=self.gate_min.val())

            if self.R_open.val() is not None:
                if (self.R_open.val() > 2e6) or (self.R_open.val() < 0):
                    self.source.failed(True)
                    self.drain.failed(True)

    def as_dataframe(self):

        jstr = self._to_json()
        jdict = parse_json_dump(jstr)
        df = pd.DataFrame([jdict])

        return df
