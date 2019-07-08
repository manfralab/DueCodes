'''
This module contains simple sweeps for measurements
and relies on mmplot for plotting.
'''

import time
import logging
import qcodes as qc
from qcodes.dataset.measurements import Measurement
import numpy as np
from mmplot import start_listener, listener_is_running
from mmplot.qcodes_dataset import QCSubscriber
from .drivers.parameters import TimerParam

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

        plot_subscriber = QCSubscriber(ds.dataset, param_set, param_meas,
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
        raise ValueError('xarray is not monotonic. This is going to break mmplot.')

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

def do2d(param_set1, xarray, delay1,
         param_set2, yarray, delay2,
         *param_meas, send_grid=True, plot_logs=False, write_period=0.1):

    if not is_monotonic(xarray):
        raise ValueError('xarray is not monotonic. This is going to break mmplot.')
    elif not is_monotonic(yarray):
        raise ValueError('yarray is not monotonic. This is going to break mmplot.')

    if not listener_is_running():
        start_listener()

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_set1)
    param_set1.post_delay = 0
    meas.register_parameter(param_set2)
    param_set2.post_delay = 0

    output = []
    for parameter in param_meas:
        meas.register_parameter(parameter, setpoints=(param_set1,param_set2))
        output.append([parameter, None])

    with meas.run() as ds:

        if send_grid:
            grid = [xarray, yarray]
        else:
            grid = None

        plot_subscriber = QCSubscriber(ds.dataset, [param_set1, param_set2], param_meas,
                                           grid=grid, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        for y in yarray:
            param_set2.set(y)
            time.sleep(delay2)
            for x in xarray:
                param_set1.set(x)
                time.sleep(delay1)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                ds.add_result((param_set1, x),
                                     (param_set2, y),
                                     *output)
        time.sleep(write_period) # let final data points propogate to plot

    return ds.run_id  # convenient to have for plotting

###################
### SPECIALIZED ###
###################

def gate_leak_check(volt_set, volt_limit, volt_step, delay, curr_meas, curr_limit,
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

    meas.register_parameter(curr_meas, setpoints=(volt_set,))

    with meas.run() as ds:

        plot_subscriber = QCSubscriber(ds.dataset, volt_set, curr_meas,
                                       grid=None, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        for vg in np.arange(0, volt_limit, volt_step):
            volt_set.set(vg)
            time.sleep(delay)
            ileak = curr_meas.get()
            ds.add_result((volt_set, vg),
                          (curr_meas, curr_meas.get()))
            if np.abs(ileak) > curr_limit:
                vmax = vg-volt_step # previous step was the limit
                break
            else:
                vmax = vg

        for vg in np.arange(vmax, -1*volt_limit, -1*volt_step):
            volt_set.set(vg)
            time.sleep(delay)
            ileak = curr_meas.get()
            ds.add_result((volt_set, vg),
                          (curr_meas, curr_meas.get()))
            if np.abs(ileak) > curr_limit:
                vmin = vg+volt_step # previous step was the limit
                break
            else:
                vmin = vg

        time.sleep(write_period) # let final data points propogate to plot

    volt_set.set(0.0)

    return vmin, vmax # convenient to have for plotting
