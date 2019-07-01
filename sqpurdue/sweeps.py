'''
This module contains simple sweeps for measurements
and relies on mmplot for plotting.
'''

import time
import qcodes as qc
from qcodes.dataset.measurements import Measurement
import numpy as np
from mmplot import start_listener, listener_is_running
from mmplot.qcodes_dataset import QCSubscriber

log = logging.getLogger(__name__)

def is_monotonic(a):
    return (np.all(np.diff(a) > 0) or np.all(np.diff(a) < 0))

class CounterParam(qc.Parameter):
    def __init__(self, name):
        # only name is required
        super().__init__(name, label='Times this has been read',
                         vals=qc.validators.Ints(min_value=0),
                         docstring='counts how many times get has been called '
                                   'but can be reset to any integer >= 0 by set')
        self._count = 0

    def get_raw(self):
        out = self._count
        self._count += 1
        return out

    def set_raw(self, val):
        self._count = val

class TimerParam(qc.Parameter):
    def __init__(self, name):
        # only name is required
        super().__init__(name, label='time elapsed since __init__() or reset()',
                         docstring='number of seconds elapsed from the \
                                    last call to TimerParam.__init__ or \
                                    TimerParam.reset()')

        self.tstart = time.time()

    def get_raw(self):
        return time.time() - self.tstart

def do1d(param_set, xarray, delay, *param_meas,
         send_grid=True, plot_logs=False, write_period=0.1):

    if not is_monotonic(xarray):
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
