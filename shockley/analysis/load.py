from typing import Optional, Collection, Union
import numpy as np
import pandas as pd
from qcodes import load_by_id, load_by_guid
from qcodes.dataset.guids import validate_guid_format


def get_dataset_by_identifier(identifier: Union[int, str]):
    '''
        returns a dataset for a given identifier

        identifier (str or int): run_id or guid
    '''
    if isinstance(identifier, int):
        dataset = load_by_id(identifier)
    elif isinstance(identifier, str):
        validate_guid_format(identifier)
        dataset = load_by_guid(identifier)

    return dataset


def list_measured_params(identifier):
    dataset = get_dataset_by_identifier(identifier)
    dependent_param_list = [param.name for param in dataset.dependent_parameters]
    return dependent_param_list


def xy_to_meshgrid(xrow, yrow):
    # stolen from qcodes.dataset.plotting
    # we use a general edge calculator,
    # in the case of non-equidistantly spaced data
    # TODO: is this appropriate for a log ax?

    dxs = np.diff(xrow)/2
    dys = np.diff(yrow)/2

    x_edges = np.concatenate((np.array([xrow[0] - dxs[0]]),
                              xrow[:-1] + dxs,
                              np.array([xrow[-1] + dxs[-1]])))
    y_edges = np.concatenate((np.array([yrow[0] - dys[0]]),
                              yrow[:-1] + dys,
                              np.array([yrow[-1] + dys[-1]])))

    return np.meshgrid(x_edges, y_edges)


def dataframe_to_arrays(
    dataframe: pd.DataFrame,
    dependent_params: Collection[str] = None,
    meshgrid: bool = False
):

    if dependent_params is None:
        dependent_params = list(dataframe.columns)

    setpoints = tuple(
        level.values for level in dataframe.index.levels
    )
    data_shape = tuple(len(array) for array in setpoints)

    if len(data_shape) == 1:
        data = [
            dataframe[param].values for param in dependent_params
        ]
    elif len(data_shape) == 2:
        data_shape = data_shape[::-1] # feels wrong. looks right.
        data = [
            dataframe[param].values.reshape(data_shape) for param in dependent_params
        ]
        if meshgrid:
            setpoints = tuple(xy_to_meshgrid(*setpoints))
    else:
        raise NotImplementedError('Cannot convert to numpy array if index has more than 2 levels')

    return (*setpoints, data)


def get_data_as_dataframe(
    identifier: Union[int, str],
    dependent_params: Collection[str] = None
):

    dataset = get_dataset_by_identifier(identifier) # load dataset

    # get/check list of measured parameters
    if dependent_params is None:
        dependent_params = [param.name for param in dataset.dependent_parameters]
    assert len(dependent_params) > 0 # need to have some measurements to look at

    # get list of sweep parameters for each measured parameter
    indpendent_params = []
    for i, param in enumerate(dependent_params):
        param_spec = dataset.paramspecs[param]
        indpendent_params.append(tuple(param_spec.depends_on_))

    # check that they are all the same
    # raises an error if more than one set of independent parameters is found
    # for different measured parameters
    assert len(set(indpendent_params)) == 1
    indpendent_params = indpendent_params[0]
    assert len(indpendent_params) > 0 # need to hvae some setpoints for data

    # create data frame with multi index
    setpoint_dict = dataset.get_setpoints(dependent_params[0])
    setpoints = tuple(
        np.array(vals).flatten() for vals in setpoint_dict.values()
    )
    multi_index = pd.MultiIndex.from_tuples(
        zip(*setpoints), names=setpoint_dict.keys()
    )
    data = {}
    for param in dependent_params:
        vals = dataset.get_values(param)
        data[param] = np.array(vals).flatten()
    return pd.DataFrame(data, index=multi_index)


def get_data_as_arrays(
    identifier: Union[int, str],
    dependent_params: Collection[str] = None,
    meshgrid: bool = False
):

    df = get_data_as_dataframe(identifier, dependent_params)

    return dataframe_to_arrays(df, meshgrid=meshgrid)
