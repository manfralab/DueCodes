'''
This module contains simple sweeps for measurements
and relies on sqpplot for plotting.
'''

import time
import logging
import qcodes as qc
from qcodes.dataset.measurements import Measurement
import numpy as np
from shockplot import start_listener, listener_is_running
from shockplot.qcodes_dataset import QCSubscriber
from .drivers.parameters import TimerParam, CounterParam

log = logging.getLogger(__name__)

def is_monotonic(a):
    return (np.all(np.diff(a) > 0) or np.all(np.diff(a) < 0))


############
### TIME ###
############

def readvstime(delay, timeout, *param_meas, plot_logs=False, write_period=0.10):
    meas = Measurement()
    meas.write_period = write_period

    timer = TimerParam('time')
    meas.register_parameter(timer)

    output = []
    for pm in param_meas:
        meas.register_parameter(pm, setpoints=(timer,))
        output.append([pm, None])

    with meas.run() as ds:

        plot_subscriber = QCSubscriber(ds.dataset, timer, param_meas,
                                       grid=None, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        while True:
            time.sleep(delay)
            t = timer.get()
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()

            ds.add_result((timer, t),
                                 *output)
            if(t>timeout):
                break
        time.sleep(write_period) # let final data points propogate to plot

    return ds.run_id  # convenient to have for plotting

############
### doND ###
############

def do1d(param_set, xarray, delay, *param_meas,
         send_grid=True, plot_logs=False, write_period=0.1):

    if not is_monotonic(xarray) and send_grid==True:
        raise ValueError('xarray is not monotonic. This is going to break shockplot.')

    if not listener_is_running():
        start_listener()

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_set)
    param_set.post_delay = 0

    output = []
    for pm in param_meas:
        meas.register_parameter(pm, setpoints=(param_set,))
        output.append([pm, None])

    with meas.run() as ds:

        if send_grid:
            grid = [xarray]
        else:
            grid = None

        plot_subscriber = QCSubscriber(ds.dataset, param_set, param_meas,
                                       grid=grid, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        param_set.set(xarray[0])
        time.sleep(0.5)

        for x in xarray:
            param_set.set(x)
            time.sleep(delay)
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()

            ds.add_result((param_set, x),
                                 *output)

        time.sleep(write_period) # let final data points propogate to plot

    return ds.run_id  # convenient to have for plotting

# do1d repeat 1 way
def do1d_repeat_oneway(param_setx, xarray, delayx, num_repeats, delayy, *param_meas,
                       send_grid=True, plot_logs=False, write_period=0.1):

    if not is_monotonic(xarray):
        raise ValueError('xarray is not monotonic. This is going to break shockplot.')

    if not listener_is_running():
        start_listener()

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_setx)
    param_setx.post_delay = 0
    param_county = CounterParam('repeat')
    meas.register_parameter(param_county)

    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_setx, param_county))
        output.append([parameter, None])

    with meas.run() as ds:

        if send_grid:
            grid = [xarray, np.arange(num_repeats)]
        else:
            grid = None

        plot_subscriber = QCSubscriber(ds.dataset, [param_setx, param_county], param_meas,
                                           grid=grid, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        for i in range(num_repeats):

            y = param_county.get()
            param_setx.set(xarray[0])
            time.sleep(delayy)
            for x in xarray:
                param_setx.set(x)
                time.sleep(delayx)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                ds.add_result((param_setx, x),
                              (param_county, y),
                              *output)
        time.sleep(write_period) # let final data points propogate to plot

    return ds.run_id  # convenient to have for plotting

def do1d_repeat_twoway(param_setx, xarray, delayx, num_repeats, delayy, *param_meas,
                       send_grid=True, plot_logs=False, write_period=0.1):

    if not is_monotonic(xarray):
        raise ValueError('xarray is not monotonic. This is going to break shockplot.')

    if not listener_is_running():
        start_listener()

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_setx)
    param_setx.post_delay = 0
    param_county = CounterParam('repeat')
    meas.register_parameter(param_county)

    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_setx, param_county))
        output.append([parameter, None])

    with meas.run() as ds:

        if send_grid:
            grid = [xarray, np.arange(num_repeats)]
        else:
            grid = None

        plot_subscriber = QCSubscriber(ds.dataset, [param_setx, param_county], param_meas,
                                           grid=grid, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        for i in range(num_repeats):


            y = param_county.get()
            if y%2==0:
                xsetpoints = xarray
            else:
                xsetpoints = xarray[::-1]

            param_setx.set(xsetpoints[0])
            time.sleep(delayy)
            for x in xsetpoints:
                param_setx.set(x)
                time.sleep(delayx)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                ds.add_result((param_setx, x),
                              (param_county, y),
                              *output)
        time.sleep(write_period) # let final data points propogate to plot

    return ds.run_id  # convenient to have for plotting

def do2d(param_setx, xarray, delayx,
         param_sety, yarray, delayy,
         *param_meas, send_grid=True, plot_logs=False, write_period=0.1):

    if not is_monotonic(xarray):
        raise ValueError('xarray is not monotonic. This is going to break shockplot.')
    elif not is_monotonic(yarray):
        raise ValueError('yarray is not monotonic. This is going to break shockplot.')

    if not listener_is_running():
        start_listener()

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_setx)
    param_setx.post_delay = 0
    meas.register_parameter(param_sety)
    param_sety.post_delay = 0

    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_setx, param_sety))
        output.append([parameter, None])

    with meas.run() as ds:

        if send_grid:
            grid = [xarray, yarray]
        else:
            grid = None

        plot_subscriber = QCSubscriber(ds.dataset, [param_setx, param_sety], param_meas,
                                           grid=grid, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        for y in yarray:
            param_sety.set(y)
            param_setx.set(x[0])
            time.sleep(delayy)
            for x in xarray:
                param_setx.set(x)
                time.sleep(delayx)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                ds.add_result((param_setx, x),
                              (param_sety, y),
                              *output)

        time.sleep(write_period) # let final data points propogate to plot

    return ds.run_id  # convenient to have for plotting

###################
### SPECIALIZED ###
###################

def gate_leak_check(volt_set, volt_limit, volt_step, delay, curr_dc,
                    curr_limit, compliance=1e-9,
                    plot_logs=False, write_period=0.1):

    if not listener_is_running():
        start_listener()

    volt_limit = np.abs(volt_limit) # should be positive
    volt_step = np.abs(volt_step) # same
    curr_limit = np.abs(curr_limit) # one more

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(volt_set)
    volt_set.post_delay = 0

    param_meas = [curr_dc]

    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(volt_set,))
        output.append([parameter, None])

    with meas.run() as ds:

        plot_subscriber = QCSubscriber(ds.dataset, volt_set, param_meas,
                                       grid=None, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        curr = np.array([])
        for vg in np.arange(0, volt_limit+volt_step, volt_step):
            volt_set.set(vg)
            time.sleep(delay)

            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            ds.add_result((volt_set, vg),
                          *output)

            curr = np.append(curr, output[0][1])
            if vg>0:
                curr = np.append(curr, output[0][1])
                if curr.max() - curr.min() > curr_limit:
                    vmax = vg-volt_step # previous step was the limit
                    curr = curr[:-1] # drop the offending element
                    break
                elif np.abs(curr[-1])>compliance:
                    vmax = vg-volt_step # previous step was the limit
                    curr = curr[:-1] # drop the offending element
                    break
                else:
                    vmax = vg

        curr = np.array([])
        for vg in np.arange(vmax, -1*volt_limit-volt_step, -1*volt_step):
            volt_set.set(vg)
            time.sleep(delay)

            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            ds.add_result((volt_set, vg),
                          *output)

            if vg<0:
                curr = np.append(curr, output[0][1])
                if curr.max() - curr.min() > curr_limit:
                    vmin = vg+volt_step # previous step was the limit
                    break
                elif np.abs(curr[-1])>compliance:
                    vmin = vg+volt_step # previous step was the limit
                    curr = curr[:-1] # drop the offending element
                    break
                else:
                    vmin = vg

        for vg in np.arange(vmin, 0.0+volt_step, volt_step):
            volt_set.set(vg)
            time.sleep(delay)

            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()
            ds.add_result((volt_set, vg),
                          *output)

        time.sleep(write_period) # let final data points propogate to plot

    return vmin, vmax # convenient to have for plotting

def check_pinchoff(gate_set, xarray, delay, curr_param,
                   v_bias = 0.01, max_allowed_r = 1e7,
                   send_grid=True, plot_logs=False, write_period=0.1):

    if not is_monotonic(xarray) and send_grid==True:
        raise ValueError('xarray is not monotonic. This is going to break shockplot.')

    if not listener_is_running():
        start_listener()

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(gate_set)
    gate_set.post_delay = 0
    meas.register_parameter(curr_param, setpoints=(gate_set,))

    with meas.run() as ds:

        if send_grid:
            grid = [xarray]
        else:
            grid = None

        plot_subscriber = QCSubscriber(ds.dataset, gate_set, [curr_param],
                                       grid=grid, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        gate_set.set(xarray[0])
        time.sleep(0.25)
        _ = curr_param.get()
        time.sleep(0.25)

        current = np.full(len(xarray), np.nan, dtype=np.float)
        v_stop = None
        for i,x in enumerate(xarray):

            gate_set.set(x)
            time.sleep(delay)

            current[i] = curr_param.get()
            ds.add_result((gate_set, x),
                          (curr_param, current[i]))

            if i<10:
                continue
            elif i==10:
                r_open = v_bias/np.abs(np.nanmean(current))/1e3
                print(f'R_open = {r_open:.2f}kOhms')
                if r_open>max_allowed_r:
                    print(f'EXITING: R_open = {r_open} greater than limit ({max_allowed_r})')
                    break
            else:
                # check if pinched off
                current_now = np.nanmean(current[i-10:i])
                std_now = np.nanstd(current[i-10:i])
                if np.abs(current_now)<std_now:
                    if v_stop is None:
                        v_stop=x-0.5
            if (v_stop is not None) and (x < v_stop):
                print(f'EXITING: device looks to be pinched off')
                break

        time.sleep(write_period) # let final data points propogate to plot

    return ds.run_id  # convenient to have for plotting
