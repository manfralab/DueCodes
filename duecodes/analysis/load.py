from typing import Optional, List, Union
import numpy as np
import pandas as pd
import xarray as xr
from qcodes import load_by_id, load_by_guid
from qcodes.dataset.guids import validate_guid_format
from .math import xy_to_meshgrid


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


def dataframe_to_xarray(df):
    """
    Convert pandas DataFrame with MultiIndex to an xarray DataSet.
    """

    arr = xr.Dataset(df)
    arr = arr.unstack('dim_0')

    return arr.transpose() # axes ordered fast -> slow (most of the time)


def get_data_as_xarray(
    identifier: Union[int, str],
    dependent_params: List[str] = None
):

    df = get_data_as_dataframe(
        identifier,
        dependent_params=dependent_params
    )
    return dataframe_to_xarray(df)


def get_data_as_dataframe(
    identifier: Union[int, str],
    dependent_params: List[str] = None
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
