"""
This module contains simple sweeps for measurements
"""

import time
from typing import (Callable, Iterator, List, Optional,
                    Sequence, Tuple, Union, Dict)
from warnings import warn
import numpy as np
import pandas as pd
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
    exp: Experiment = None,
    use_threads=False,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    additional_setpoints = tuple(),
):

    meas = Measurement(exp=exp)

    timer = ElapsedTimeParameter("time")

    all_setpoint_params = (timer,) + tuple(
        s for s in additional_setpoints)

    measured_parameters = list(param for param in param_meas
                                if isinstance(param, _BaseParameter))


    if (len(measured_parameters)>2) or (use_threads==True):
        use_threads = True
    else:
        use_threads = False

    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params, shapes=None)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        timer.reset_clock()

        while True:
            time.sleep(delay)
            datasaver.add_result(
                (timer, timer.get()),
                *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                *additional_setpoints_data
            )
            if (timeout - timer.get()) < 0.005:
                break
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
    use_threads: Optional[bool] = None,
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

    if (len(measured_parameters)>2) or (use_threads==True):
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
    qcnd._register_actions(meas, enter_actions, exit_actions)

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

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
    exp: Optional[Experiment] = None,
    use_threads: Optional[bool] = None,
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

    if (len(measured_parameters)>2) or (use_threads==True):
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
    qcnd._register_actions(meas, enter_actions, exit_actions)

    param_setx.post_delay = 0.0

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

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
    exp: Optional[Experiment] = None,
    use_threads: Optional[bool] = None,
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

    if (len(measured_parameters)>2) or (use_threads==True):
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
    qcnd._register_actions(meas, enter_actions, exit_actions)

    param_setx.post_delay = 0.0

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

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


# def do2d_twoway(
#     param_setx, xarray, delayx,
#     param_sety, yarray, delayy,
#     *param_meas: qcnd.ParamMeasT,
#     enter_actions: qcnd.ActionsT = (),
#     exit_actions: qcnd.ActionsT = (),
#     before_inner_actions: qcnd.ActionsT = (),
#     after_inner_actions: qcnd.ActionsT = (),
#     exp: Optional[Experiment] = None,
#     use_threads: Optional[bool] = None,
#     additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
# ):

#     meas_fwd = Measurement(exp=exp)
#     meas_rev = Measurement(exp=exp)

#     all_setpoint_params = (param_sety, param_setx,) + tuple(
#             s for s in additional_setpoints)

#     measured_parameters = tuple(param for param in param_meas
#                                 if isinstance(param, _BaseParameter))

#     if (len(measured_parameters)>2) or (use_threads==True):
#         use_threads = True
#     else:
#         use_threads = False

#     try:
#         loop_shape = tuple(
#             1 for _ in additional_setpoints
#         ) + (len(yarray), len(xarray))
#         shapes: Shapes = detect_shape_of_measurement(
#             measured_parameters,
#             loop_shape
#         )
#     except TypeError:
#         warn(
#             f"Could not detect shape of {measured_parameters} "
#             f"falling back to unknown shape.")
#         shapes = None

#     # forward dataset
#     qcnd._register_parameters(meas_fwd, all_setpoint_params)
#     qcnd._register_parameters(meas_fwd, param_meas, setpoints=all_setpoint_params,
#                          shapes=shapes)
#     qcnd._register_actions(meas_fwd, enter_actions, exit_actions)

#     # reversed dataset
#     qcnd._register_parameters(meas_rev, all_setpoint_params)
#     qcnd._register_parameters(meas_rev, param_meas, setpoints=all_setpoint_params,
#                          shapes=shapes)
#     qcnd._register_actions(meas_rev, enter_actions, exit_actions)

#     param_setx.post_delay = 0.0
#     param_sety.post_delay = 0.0

#     with qcnd._catch_keyboard_interrupts() as interrupted, \
#         meas_fwd.run() as ds_fwd, \
#         meas_rev.run() as ds_rev:

#         print(f"forward sweeps: {ds_fwd.run_id}, reversed sweeps: {ds_rev.run_id}")

#         additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
#         for j, set_pointy in enumerate(yarray):

#             if j % 2 == 0:
#                 xsetpoints = xarray
#                 datasaver = ds_fwd
#             else:
#                 xsetpoints = xarray[::-1]
#                 datasaver = ds_rev
#             param_sety.set(set_pointy)
#             param_setx.set(xsetpoints[0])
#             time.sleep(delayy)

#             for action in before_inner_actions:
#                 action()
#             for set_pointx in xsetpoints:
#                 param_setx.set(set_pointx)
#                 time.sleep(delayx)

#                 datasaver.add_result((param_sety, set_pointy),
#                                      (param_setx, set_pointx),
#                                      *qcnd._process_params_meas(param_meas, use_threads=use_threads),
#                                      *additional_setpoints_data)

#             for action in after_inner_actions:
#                 action()

#     return ds_fwd.dataset, ds_rev.dataset

def do2d(
    param_setx, xarray, delayx,
    param_sety, yarray, delayy,
    *param_meas: qcnd.ParamMeasT,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    before_inner_actions: qcnd.ActionsT = (),
    after_inner_actions: qcnd.ActionsT = (),
    exp: Optional[Experiment] = None,
    use_threads: Optional[bool] = None,
    additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
):

    meas = Measurement(exp=exp)
    all_setpoint_params = (param_sety, param_setx,) + tuple(
            s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))

    if (len(measured_parameters)>2) or (use_threads==True):
        use_threads = True
    elif (use_threads==False):
        use_threads = False
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
    qcnd._register_actions(meas, enter_actions, exit_actions)

    param_setx.post_delay = 0.0
    param_sety.post_delay = 0.0

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        for set_pointy in yarray:
            param_setx.set(xarray[0])
            param_sety.set(set_pointy)
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

# def do2d_two_inner(
#     param_setx_1, xarray_1, delayx_1,
#     param_setx_2, xarray_2, delayx_2,
#     param_sety, yarray, delayy,
#     param_meas_1: Sequence[qcnd.ParamMeasT],
#     param_meas_2: Sequence[qcnd.ParamMeasT],
#     enter_actions: qcnd.ActionsT = (),
#     exit_actions: qcnd.ActionsT = (),
#     before_inner_actions: qcnd.ActionsT = (),
#     after_inner_actions: qcnd.ActionsT = (),
#     exp: Optional[Experiment] = None,
#     use_threads: Optional[bool] = None,
#     additional_setpoints: Sequence[qcnd.ParamMeasT] = tuple(),
# ):
#
#     meas_1 = Measurement(exp=exp)
#     meas_2 = Measurement(exp=exp)
#
#     all_setpoint_params_1 = (param_sety, param_setx_1,) + tuple(
#             s for s in additional_setpoints)
#     all_setpoint_params_2 = (param_sety, param_setx_2,) + tuple(
#             s for s in additional_setpoints)
#
#     measured_parameters_1 = tuple(param for param in param_meas_1
#                                   if isinstance(param, _BaseParameter))
#     measured_parameters_2 = tuple(param for param in param_meas_2
#                                   if isinstance(param, _BaseParameter))
#
#     if (len(measured_parameters_1)>2) or (len(measured_parameters_2)>2) or (use_threads==True):
#         use_threads = True
#     elif (use_threads==False):
#         use_threads = False
#     else:
#         use_threads = False
#
#     try:
#         loop_shape_1 = tuple(
#             1 for _ in additional_setpoints
#         ) + (len(yarray), len(xarray_1))
#         shapes_1: Shapes = detect_shape_of_measurement(
#             measured_parameters_1,
#             loop_shape_1
#         )
#     except TypeError:
#         warn(
#             f"Could not detect shape of {measured_parameters} "
#             f"falling back to unknown shape.")
#         shapes = None
#
#     try:
#         loop_shape_2 = tuple(
#             1 for _ in additional_setpoints
#         ) + (len(yarray), len(xarray_2))
#         shapes_2: Shapes = detect_shape_of_measurement(
#             measured_parameters_2,
#             loop_shape_2
#         )
#     except TypeError:
#         warn(
#             f"Could not detect shape of {measured_parameters} "
#             f"falling back to unknown shape.")
#         shapes = None
#
#
#     qcnd._register_parameters(meas_1, all_setpoint_params_1)
#     qcnd._register_parameters(meas_1, param_meas_1, setpoints=all_setpoint_params_1,
#                          shapes=shapes_1)
#     qcnd._register_actions(meas_1, enter_actions, exit_actions)
#
#     qcnd._register_parameters(meas_2, all_setpoint_params_2)
#     qcnd._register_parameters(meas_2, param_meas_2, setpoints=all_setpoint_params_2,
#                          shapes=shapes_2)
#
#     param_setx_1.post_delay = 0.0
#     param_setx_2.post_delay = 0.0
#     param_sety.post_delay = 0.0
#
#     with qcnd._catch_keyboard_interrupts() as interrupted, \
#         meas_1.run(write_in_background=True) as datasaver_1, \
#         meas_2.run(write_in_background=True) as datasaver_2:
#
#         additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
#         for set_pointy in yarray:
#             for action in before_inner_actions:
#                 action()
#
#             param_setx_1.set(xarray_1[0])
#             param_setx_2.set(xarray_2[0])
#             param_sety.set(set_pointy)
#             time.sleep(delayy)
#
#             for set_pointx in xarray_1:
#                 param_setx_1.set(set_pointx)
#                 time.sleep(delayx_1)
#
#                 datasaver_1.add_result((param_sety, set_pointy),
#                                      (param_setx_1, set_pointx),
#                                      *qcnd._process_params_meas(param_meas_1, use_threads=use_threads),
#                                      *additional_setpoints_data)
#
#             param_setx_1.set(xarray_1[0])
#             param_setx_2.set(xarray_2[0])
#             param_sety.set(set_pointy)
#             time.sleep(delayy)
#             for set_pointx in xarray_2:
#                 param_setx_2.set(set_pointx)
#                 time.sleep(delayx_2)
#
#                 datasaver_2.add_result((param_sety, set_pointy),
#                                      (param_setx_2, set_pointx),
#                                      *qcnd._process_params_meas(param_meas_2, use_threads=use_threads),
#                                      *additional_setpoints_data)
#
#
#             for action in after_inner_actions:
#                 action()
#
#         dataset_1 = datasaver_1.dataset
#         dataset_2 = datasaver_2.dataset
#     return dataset_1, dataset_2

### AMI MAGNET

def field_sweep_ami(
    field_param,
    start: float,
    stop: float,
    delay: float,
    *param_meas: qcnd.ParamMeasT,
    exp: Experiment = None,
    use_threads=False,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    additional_setpoints = tuple(),
):

    # get instrument for field param
    magnet = field_param.instrument

    # timer param
    timer = ElapsedTimeParameter("time")

    # add field to measured params
    param_meas = list(param_meas)
    param_meas.append(field_param)
    measured_parameters = list(param for param in param_meas
                            if isinstance(param, _BaseParameter))

    all_setpoint_params = (timer,) + tuple(
        s for s in additional_setpoints)

    if (len(measured_parameters)>2) or (use_threads==True):
        use_threads = True
    elif (use_threads==False):
        use_threads = False
    else:
        use_threads = False

    meas = Measurement(exp=exp)
    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params, shapes=None)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

        magnet.set_field(start, block=True)
        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        timer.reset_clock()
        magnet.set_field(stop, block=False)

        while magnet.ramping_state() == 'ramping':
            time.sleep(delay)
            datasaver.add_result(
                (timer, timer.get()),
                *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                *additional_setpoints_data
            )

        dataset = datasaver.dataset

    return dataset

def _bin_results_to_fit_shape(param_setx, xrow, all_params, results_dict):
    # results_dict contacts param_name: [list of values]
    # for all parameters measured during inner loop
    # inner lists contain all results at a given time
    # param_setx will be used as the set point
    # xarray will be used to bin results according to setpoint values
    # returns data

    dxs = np.diff(xrow)/2
    bins = np.concatenate((np.array([xrow[0] - dxs[0]]),
                              xrow[:-1] + dxs,
                              np.array([xrow[-1] + dxs[-1]])))

    df = pd.DataFrame(results_dict)
    df = df.groupby(pd.cut(df[param_setx.name], bins)).mean(0)
    df = df.set_index(param_setx.name)

    output = [(param_setx, xrow)]
    for col in df.columns:
        col_param = next(p for p in all_params if p.name==col)
        output.append((col_param, df[col].values))

    return output

def field_sweep_ami_2d(
    field_param, xarray, delayx,
    param_sety, yarray, delayy,
    *param_meas: qcnd.ParamMeasT,
    exp: Experiment = None,
    use_threads=False,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    additional_setpoints = tuple(),
):

    # get instrument for field param
    magnet = field_param.instrument

    # add field to measured params
    all_setpoint_params = (param_sety, field_param,) + tuple(
            s for s in additional_setpoints)

    measured_parameters = tuple(param for param in param_meas
                                if isinstance(param, _BaseParameter))

    if (len(measured_parameters)>2) or (use_threads==True):
        use_threads = True
    elif (use_threads==False):
        use_threads = False
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

    meas = Measurement(exp=exp)
    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params, shapes=None)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    inner_loop_params = (field_param,) + param_meas
    inner_loop_dict = {k.name: [] for k in inner_loop_params}

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)

        for set_pointy in yarray:
            param_sety.set(set_pointy)
            time.sleep(delayy)

            magnet.set_field(xarray[0], block=True)
            magnet.set_field(xarray[-1], block=False)

            while magnet.ramping_state() == 'ramping':
                time.sleep(delayx)

                for param, val in qcnd._process_params_meas(inner_loop_params, use_threads=use_threads):
                    inner_loop_dict[param.name].append(val)

            datasaver.add_result(
                (param_sety, [set_pointy]*len(xarray)),
                *_bin_results_to_fit_shape(field_param, xarray, inner_loop_params, inner_loop_dict),
                *additional_setpoints_data,
            )

        dataset = datasaver.dataset

    return dataset


### PS120 MAGNET ###

def field_sweep_ps120(
    field_param,
    start: float,
    stop: float,
    delay: float,
    *param_meas: qcnd.ParamMeasT,
    exp: Experiment = None,
    use_threads=False,
    enter_actions: qcnd.ActionsT = (),
    exit_actions: qcnd.ActionsT = (),
    additional_setpoints = tuple(),
):

    # get instrument for field param
    magnet = field_param.instrument

    # timer param
    timer = ElapsedTimeParameter("time")

    # add field to measured params
    param_meas = list(param_meas)
    param_meas.append(field_param)
    measured_parameters = list(param for param in param_meas
                            if isinstance(param, _BaseParameter))

    all_setpoint_params = (timer,) + tuple(
        s for s in additional_setpoints)

    if (len(measured_parameters)>2) or (use_threads==True):
        use_threads = True
    elif (use_threads==False):
        use_threads = False
    else:
        use_threads = False

    meas = Measurement(exp=exp)
    qcnd._register_parameters(meas, all_setpoint_params)
    qcnd._register_parameters(meas, param_meas, setpoints=all_setpoint_params, shapes=None)
    qcnd._register_actions(meas, enter_actions, exit_actions)

    with qcnd._catch_keyboard_interrupts() as interrupted, \
        meas.run(write_in_background=True) as datasaver:

        magnet.field_blocking(start)
        additional_setpoints_data = qcnd._process_params_meas(additional_setpoints)
        timer.reset_clock()
        magnet.field_non_blocking(stop)

        while magnet.ramp_status() != magnet._GET_STATUS_RAMP[0]:
            time.sleep(delay)
            datasaver.add_result(
                (timer, timer.get()),
                *qcnd._process_params_meas(param_meas, use_threads=use_threads),
                *additional_setpoints_data
            )

        dataset = datasaver.dataset

    return dataset
