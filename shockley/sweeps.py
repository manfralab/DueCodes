'''
This module contains simple sweeps for measurements
and relies on sqpplot for plotting.
'''

import time
import logging
import numpy as np
from qcodes.dataset.measurements import Measurement
from qchart import start_listener, listener_is_running
from shockley.drivers.parameters import TimerParam, CounterParam

LOGGER = logging.getLogger(__name__)

def is_monotonic(arr):
    """ check if array is monotonic """
    return np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)

def gen_sweep_array(start, stop, step=None, num=None):
    """
    Generate numbers over a specified interval.
    Requires `start` and `stop` and (`step` or `num`)
    The sign of `step` is not relevant.
    Args:
        start (Union[int, float]): The starting value of the sequence.
        stop (Union[int, float]): The end value of the sequence.
        step (Optional[Union[int, float]]):  Spacing between values.
        num (Optional[int]): Number of values to generate.
    Returns:
        numpy.ndarray: numbers over a specified interval as a ``numpy.linspace``
    Examples:
        >>> make_sweep(0, 10, num=5)
        [0.0, 2.5, 5.0, 7.5, 10.0]
        >>> make_sweep(5, 10, step=1)
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        >>> make_sweep(15, 10.5, step=1.5)
        >[15.0, 13.5, 12.0, 10.5]
    """
    if step and num:
        raise AttributeError('Don\'t use `step` and `num` at the same time.')
    if (step is None) and (num is None):
        raise ValueError('If you really want to go from `start` to '
                         '`stop` in one step, specify `num=2`.')
    if step is not None:
        steps = abs((stop - start) / step)
        tolerance = 1e-10
        steps_lo = int(np.floor(steps + tolerance))
        steps_hi = int(np.ceil(steps - tolerance))

        if steps_lo != steps_hi:
            real_step = abs((stop-start) / (steps_lo+1))
            if abs(step - real_step)/step > 0.05:
                # print a warning if the effective step size is more than 2% d
                # different than what was requested
                print('WARNING: Could not find an integer number of points for '
                      'the the given `start`, `stop`, and `step`={0}.'
                      ' Effective step size is `step`={1:.4f}'.format(step, real_step))
        num = steps_lo + 1

    return np.linspace(start, stop, num=num)


############
### TIME ###
############


def readvstime(
    delay, timeout, *param_meas,
    plot_logs=False, write_period=0.50
):

    if not listener_is_running():
        start_listener()
        time.sleep(0.25)

    meas = Measurement()
    meas.write_period = write_period

    timer = TimerParam('time')
    meas.register_parameter(timer)

    output = []
    for pm in param_meas:
        meas.register_parameter(pm, setpoints=(timer,))
        output.append([pm, None])

    with meas.run() as ds:

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

    return ds


############
### doND ###
############


def do1d(
    param_set, xarray, delay, *param_meas,
    plot_logs=False, write_period=0.50
):

    if not listener_is_running():
        start_listener()
        time.sleep(0.25)

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_set)
    param_set.post_delay = 0

    output = []
    for pm in param_meas:
        meas.register_parameter(pm, setpoints=(param_set,))
        output.append([pm, None])

    with meas.run() as ds:

        param_set.set(xarray[0])
        time.sleep(0.5)

        for x in xarray:
            param_set.set(x)
            time.sleep(delay)
            for i, parameter in enumerate(param_meas):
                output[i][1] = parameter.get()

            ds.add_result(
                (param_set, x),
                *output
            )

        time.sleep(write_period) # let final data points propogate to plot

    return ds


def do1d_repeat_oneway(
    param_setx, xarray, delayx, num_repeats, delayy, *param_meas,
    plot_logs=False, write_period=0.50
):

    if not listener_is_running():
        start_listener()
        time.sleep(0.25)

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

        for i in range(num_repeats):

            y = param_county.get()
            param_setx.set(xarray[0])
            time.sleep(delayy)
            for x in xarray:
                param_setx.set(x)
                time.sleep(delayx)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                ds.add_result(
                    (param_setx, x),
                    (param_county, y),
                    *output
                )
        time.sleep(write_period) # let final data points propogate to plot

    return ds


def do1d_repeat_twoway(
    param_setx, xarray, delayx, num_repeats, delayy, *param_meas,
    plot_logs=False, write_period=0.50
):

    if not listener_is_running():
        start_listener()
        time.sleep(0.25)

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_setx)
    param_setx.post_delay = 0
    param_county = CounterParam('repeat')
    meas.register_parameter(param_county)

    output = []
    for parameter in param_meas:
        meas.register_parameter(
            parameter, setpoints=(param_setx, param_county)
        )
        output.append([parameter, None])

    with meas.run() as ds:

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
                ds.add_result(
                    (param_setx, x),
                    (param_county, y),
                    *output
                )
        time.sleep(write_period) # let final data points propogate to plot

    return ds

def do2d(
    param_setx, xarray, delayx,
    param_sety, yarray, delayy,
    *param_meas,
    plot_logs=False, write_period=0.50
):

    if not listener_is_running():
        start_listener()
        time.sleep(0.25)

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(param_setx)
    param_setx.post_delay = 0
    meas.register_parameter(param_sety)
    param_sety.post_delay = 0

    output = []
    for parameter in param_meas:
        meas.register_parameter(
            parameter, setpoints=(param_setx, param_sety)
        )
        output.append([parameter, None])

    with meas.run() as ds:

        for y in yarray:
            param_sety.set(y)
            param_setx.set(xarray[0])
            time.sleep(delayy)
            for x in xarray:
                param_setx.set(x)
                time.sleep(delayx)
                for i, parameter in enumerate(param_meas):
                    output[i][1] = parameter.get()
                ds.add_result(
                    (param_setx, x),
                    (param_sety, y),
                    *output
                )

        time.sleep(write_period) # let final data points propogate to plot

    return ds
