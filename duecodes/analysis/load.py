from typing import Optional, List, Union
from warnings import warn
import numpy as np
import pandas as pd
from pandas import DataFrame
from xarray import Dataset
from qcodes.dataset.data_set import load_by_run_spec
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.guids import validate_guid_format


def get_dataset_by_identifier(identifier: Union[int, str]):
    """
        returns a dataset for a given identifier

        identifier (str or int): run_id or guid
    """
    if isinstance(identifier, int):
        dataset = load_by_run_spec(captured_run_id=identifier)
    elif isinstance(identifier, str):
        validate_guid_format(identifier)
        dataset = load_by_guid(identifier)

    return dataset


def _empty_dataset_dict(dataset: DataSet):
    assert (
        len(dataset.dependent_parameters) > 0
    )  # need to have some measurements to look at

    # get list of sweep parameters for each measured parameter
    dependent_params = []
    dependent_units = []
    independent_params = []
    for param in dataset.dependent_parameters:
        dependent_params.append(param.name)
        param_spec = dataset.paramspecs[param.name]
        dependent_units.append(param_spec.unit)
        independent_params.append(tuple(param_spec.depends_on_))

    # check that they are all the same
    # raises an error if more than one set of independent parameters is found
    # for different measured parameters
    assert len(set(independent_params)) == 1
    independent_params = list(independent_params[0])
    assert len(independent_params) > 0  # need to have some setpoints for data
    independent_units = []
    for param in independent_params:
        param_spec = dataset.paramspecs[param]
        independent_units.append(param_spec.unit)

    if len(independent_params) == 1:
        sweep_type = "1D"
    elif len(independent_params) == 2:
        sweep_type = "2D"
    else:
        sweep_type = "unknown"

    return {
        "Independent Parameters": list(zip(independent_params, independent_units)),
        "Dependent Parameters": list(zip(dependent_params, dependent_units)),
        "Dataset Length": dataset.number_of_results,
        "Sweep Type": sweep_type,
        "Start/End Times": (dataset.run_timestamp(), dataset.completed_timestamp()),
    }


def get_dataset_description(identifier: Union[int, str]) -> dict:

    dataset = get_dataset_by_identifier(identifier)
    dataset_dict = _empty_dataset_dict(dataset)

    return dataset_dict


def get_data_as_numpy(identifier: Union[int, str], *params: str):
    """
        returns (x, y_list) or (x, y, z_list)
    """
    xarr = get_data_as_xarray(identifier, *params)

    return (
        *list(xarr.coords[param].values for param in xarr.coords),
        list(xarr[param].values for param in xarr),
    )


def _dataframe_to_xarray(df: DataFrame) -> Dataset:
    """
    Convert pandas DataFrame with MultiIndex to an xarray DataSet.
    """

    len_old_df = len(df)
    df = df[~df.index.duplicated()]
    len_new_df = len(df)

    # if len_new_df < len_old_df:
    #     warn("Duplicate values removed from DataFrame. This dataset is weird.")

    return df.to_xarray()


def get_data_as_xarray(identifier: Union[int, str], *params: str):
    """
        identifier: guid or run_id
        *params: dependent (measured) parameters to be included in the returned DataFrame
    """

    df = get_data_as_dataframe(identifier, *params)
    return _dataframe_to_xarray(df)


def get_data_as_dataframe(identifier: Union[int, str], *params: str) -> DataFrame:
    """
        identifier: guid or run_id
        *params: dependent (measured) parameters to be included in the returned DataFrame
    """

    dataset = get_dataset_by_identifier(identifier)  # load dataset
    if params is None:
        dependent_parameters = dataset.dependent_parameters
    else:
        _check_dependent_params(dataset, *params)


    dataframe_dict = {}
    datadict = dataset.get_parameter_data(*params)
    for name, subdict in datadict.items():
        keys = list(subdict.keys())
        if len(keys) == 0:
            dataframe_dict[name] = pd.DataFrame()
            continue
        if len(keys) == 1:
            index = None
        elif len(keys) == 2:
            index = pd.Index(subdict[keys[1]].ravel(), name=keys[1])
        else:
            indexdata = tuple(np.concatenate(subdict[key])
                              if subdict[key].dtype == np.dtype('O')
                              else subdict[key].ravel()
                              for key in keys[1:])
            index = pd.MultiIndex.from_arrays(
                indexdata,
                names=keys[1:])

        if subdict[keys[0]].dtype == np.dtype('O'):
            # ravel will not fully unpack a numpy array of arrays
            # which are of "object" dtype. This can happen if a variable
            # length array is stored in the db. We use concatenate to
            # flatten these
            try:
                mydata = subdict[keys[0]].ravel()
            except ValueError:
                # not all objects are nested arrays
                mydata = subdict[keys[0]].ravel()
        else:
            mydata = subdict[keys[0]].ravel()
        df = pd.DataFrame(mydata, index=index,
                          columns=[keys[0]])
        dataframe_dict[name] = df

    return pd.concat(list(dataframe_dict.values()), axis=1)


def _check_dependent_params(dataset: DataSet, *params: str):

    dependent_parameters = [param.name for param in dataset.dependent_parameters]
    for param in params:
        if param not in dependent_parameters:
            raise TypeError(
                f"{param} is not a dependent (measured) parameter of the dataset"
            )
