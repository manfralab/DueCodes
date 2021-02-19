"""
This module contains simple sweeps for measurements
"""

import time
from typing import (Callable, Iterator, List, Optional,
                    Sequence, Tuple, Union, Dict)
from warnings import warn
import numpy as np
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.experiment_container import Experiment
from qcodes.utils.dataset import doNd as qcnd
from qcodes.instrument.parameter import _BaseParameter
from qcodes.dataset.descriptions.detect_shapes import \
    detect_shape_of_measurement
from duecodes.drivers.parameters import CountParameter, ElapsedTimeParameter


def _is_monotonic(arr):
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
        >>> gen_sweep_array(0, 10, num=5)
        [0.0, 2.5, 5.0, 7.5, 10.0]
        >>> gen_sweep_array(5, 10, step=1)
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        >>> gen_sweep_array(15, 10.5, step=1.5)
        >[15.0, 13.5, 12.0, 10.5]
    """
    if step and num:
        raise AttributeError("Don't use `step` and `num` at the same time.")
    if (step is None) and (num is None):
        raise ValueError(
            "If you really want to go from `start` to "
            "`stop` in one step, specify `num=2`."
        )
    if step is not None:
        steps = abs((stop - start) / step)
        tolerance = 1e-10
        steps_lo = int(np.floor(steps + tolerance))
        steps_hi = int(np.ceil(steps - tolerance))

        if steps_lo != steps_hi:
            real_step = abs((stop - start) / (steps_lo + 1))
            if abs(step - real_step) / step > 0.05:
                # print a warning if the effective step size is more than 2% d
                # different than what was requested
                warn(
                    "Could not find an integer number of points for "
                    "the the given `start`, `stop`, and `step`={0}."
                    " Effective step size is `step`={1:.4f}".format(step, real_step)
                )
        num = steps_lo + 1

    return np.linspace(start, stop, num=num)


############
### TIME ###
############


def readvstime(
    delay: float,
    timeout: float,
    *param_meas: qcnd.ParamMeasT,
    exp: Optional[Experiment] = None,
    background_write: bool = True,
    write_period: float = 0.25,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
):

    meas = Measurement(exp=exp)

    timer = ElapsedTimeParameter("time")

    all_setpoint_params = (timer,) + tuple(
        s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))
    if len(measured_parameters)>2:
        use_threads = True
    else:
        use_threads = False

    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params,
                         shapes=shapes)
    qcnd._set_write_period(meas, write_period)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=background_write) as datasaver:

        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        timer.reset_clock()
        while True:
            time.sleep(delay)
            datasaver.add_result(
                (timer, timer.get()),
                *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                *additional_setpoints_data
            )
        dataset = datasaver.dataset

    return dataset


############
### doND ###
############


def do1d(
    param_set: _BaseParameter,
    xarray,
    delay: float,
    *param_meas: qcnd.ParamMeasT,
    exp: Optional[Experiment] = None,
    background_write: bool = True,
    write_period: float = 0.25,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
):

    if not _is_monotonic(xarray):
        warn('Sweep array is not monotonic. This is pretty weird. Reconsider.')

    meas = Measurement(exp=exp)

    all_setpoint_params = (param_set,) + tuple(
        s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))
    if len(measured_parameters)>2:
        use_threads = True
    else:
        use_threads = False

    try:
        loop_shape = tuple(1 for _ in additional_setpoints) + (len(xarray),)
        shapes: Shapes = detect_shape_of_measurement(
            measured_parameters,
            loop_shape
        )
    except TypeError:
        warn(f"Could not detect shape of {measured_parameters} "
             f"falling back to unknown shape.")
        shapes = None

    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params,
                         shapes=shapes)
    qcnd._set_write_period(meas, write_period)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=background_write) as datasaver:

        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        for set_point in xarray:
            param_set.set(set_point)
            time.sleep(delay)
            datasaver.add_result(
                (param_set, set_point),
                *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                *additional_setpoints_data
            )
        dataset = datasaver.dataset

    return dataset


def do1d_repeat_oneway(
    param_setx, xarray, delayx,
    num_repeats, delayy,
    *param_meas: qcnd.ParamMeasT,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    before_inner_actions: qcnd.ActionsT = (),
    after_inner_actions: qcnd.ActionsT = (),
    write_period: float = 0.25,
    exp: Optional[Experiment] = None,
    additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
):
    if (not _is_monotonic(xarray)) or (not _is_monotonic(yarray)):
        warn('Sweep array is not monotonic. This is pretty weird. Reconsider.')

    meas = Measurement(exp=exp)

    param_county = CountParameter("repeat")
    all_setpoint_params = (param_county, param_setx,) + tuple(
            s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))
    if len(measured_parameters)>2:
        use_threads = True
    else:
        use_threads = False

    try:
        loop_shape = tuple(
            1 for _ in additional_setpoints
        ) + (num_repeats, len(xarray))
        shapes: Shapes = detect_shape_of_measurement(
            measured_parameters,
            loop_shape
        )
    except TypeError:
        warn(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape.")
        shapes = None

    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params,
                         shapes=shapes)
    qcnd._set_write_period(meas, write_period)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    param_setx.post_delay = 0.0

    with qcnd._catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        for i in range(num_repeats):
            y = param_county.get()
            param_setx.set(xarray[0])
            time.sleep(delayy)
            for action in before_inner_actions:
                action()
            for set_pointx in xarray:
                param_setx.set(set_pointx)
                time.sleep(delayx)

                datasaver.add_result((param_county, y),
                                     (param_setx, set_pointx),
                                     *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                                     *additional_setpoints_data)
            for action in after_inner_actions:
                action()

        dataset = datasaver.dataset
    return dataset


def do1d_repeat_twoway(
    param_setx, xarray, delayx,
    num_repeats, delayy,
    *param_meas: qcnd.ParamMeasT,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    before_inner_actions: qcnd.ActionsT = (),
    after_inner_actions: qcnd.ActionsT = (),
    write_period: float = 0.25,
    exp: Optional[Experiment] = None,
    additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
):
    if (not _is_monotonic(xarray)) or (not _is_monotonic(yarray)):
        warn('Sweep array is not monotonic. This is pretty weird. Reconsider.')

    meas = Measurement(exp=exp)

    param_county = CountParameter("repeat")
    all_setpoint_params = (param_county, param_setx,) + tuple(
            s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))
    if len(measured_parameters)>2:
        use_threads = True
    else:
        use_threads = False

    try:
        loop_shape = tuple(
            1 for _ in additional_setpoints
        ) + (num_repeats, len(xarray))
        shapes: Shapes = detect_shape_of_measurement(
            measured_parameters,
            loop_shape
        )
    except TypeError:
        warn(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape.")
        shapes = None

    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params,
                         shapes=shapes)
    qcnd._set_write_period(meas, write_period)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    param_setx.post_delay = 0.0

    with qcnd._catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        for i in range(num_repeats):
            y = param_county.get()
            if y % 2 == 0:
                xsetpoints = xarray
            else:
                xsetpoints = xarray[::-1]
            param_setx.set(xsetpoints[0])
            time.sleep(delayy)

            for action in before_inner_actions:
                action()

            for set_pointx in xsetpoints:
                param_setx.set(set_pointx)
                time.sleep(delayx)

                datasaver.add_result((param_county, y),
                                     (param_setx, set_pointx),
                                     *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                                     *additional_setpoints_data)
            for action in after_inner_actions:
                action()

        dataset = datasaver.dataset
    return dataset


def do2d(
    param_setx, xarray, delayx,
    param_sety, yarray, delayy,
    *param_meas: qcnd.ParamMeasT,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    before_inner_actions: qcnd.ActionsT = (),
    after_inner_actions: qcnd.ActionsT = (),
    write_period: float = 0.25,
    exp: Optional[Experiment] = None,
    additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
):

    meas = Measurement(exp=exp)
    all_setpoint_params = (param_sety, param_setx,) + tuple(
            s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))
    if len(measured_parameters)>2:
        use_threads = True
    else:
        use_threads = False

    try:
        loop_shape = tuple(
            1 for _ in additional_setpoints
        ) + (len(yarray), len(xarray))
        shapes: Shapes = detect_shape_of_measurement(
            measured_parameters,
            loop_shape
        )
    except TypeError:
        warn(
            f"Could not detect shape of {measured_parameters} "
            f"falling back to unknown shape.")
        shapes = None

    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params,
                         shapes=shapes)
    qcnd._set_write_period(meas, write_period)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    param_setx.post_delay = 0.0
    param_sety.post_delay = 0.0

    with qcnd._catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        for set_pointy in yarray:
            param_sety.set(set_pointy)
            param_setx.set(xarray[0])
            time.sleep(delayy)
            for action in before_inner_actions:
                action()
            for set_pointx in xarray:
                param_setx.set(set_pointx)
                time.sleep(delayx)

                datasaver.add_result((param_sety, set_pointy),
                                     (param_setx, set_pointx),
                                     *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                                     *additional_setpoints_data)

            for action in after_inner_actions:
                action()

        dataset = datasaver.dataset
    return dataset
