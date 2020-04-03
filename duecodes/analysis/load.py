from typing import Optional, List, Union
from warnings import warn
import numpy as np
import pandas as pd
from pandas import DataFrame
from xarray import Dataset
from qcodes import load_by_id, load_by_guid
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.guids import validate_guid_format
from qcodes.dataset.data_export import (
    flatten_1D_data_for_plot,
    datatype_from_setpoints_2d,
    reshape_2D_data,
)


def get_dataset_by_identifier(identifier: Union[int, str]):
    """
        returns a dataset for a given identifier

        identifier (str or int): run_id or guid
    """
    if isinstance(identifier, int):
        dataset = load_by_id(identifier)
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
        sweep_type = "1d"
    elif len(independent_params) == 2:
        sweep_type = "2d"
    else:
        sweep_type = "unknown"

    return {
        "Independent Parameters": list(zip(independent_params, independent_units)),
        "Dependent Parameters": list(zip(independent_params, independent_units)),
        "Dataset Length": dataset.number_of_results,
        "Sweep Type": sweep_type,
        "Start/End Times": (dataset.run_timestamp(), dataset.completed_timestamp()),
    }


def get_dataset_description(identifier: Union[int, str]) -> dict:

    dataset = get_dataset_by_identifier(identifier)
    dataset_dict = _empty_dataset_dict(dataset)

    return dataset_dict


# def _shaped_data_dict(dataset: DataSet, *params: str):
#
#     # setup dataset_dict
#     dataset_dict = _empty_dataset_dict(dataset)
#
#     # check params input
#     if params:
#         _check_dependent_params(dataset_dict["dependent_parameters"]["names"], *params)
#     else:
#         params = tuple(dataset_dict["dependent_parameters"]["names"])
#
#     # get all requested data from database
#     flat_data_dict = {}
#     for data_dict in dataset.get_parameter_data(*params).values():
#         for param, data in data_dict.items():
#             flat_data_dict[param] = flatten_1D_data_for_plot(data)
#
#     # store data into dataset_dict
#     for param_type in ["independent_parameters", "dependent_parameters"]:
#         for idx, param in enumerate(dataset_dict[param_type]["names"]):
#             if param in flat_data_dict:
#                 dataset_dict[param_type]["data"][idx] = flat_data_dict[param]
#                 dataset_dict[param_type]["shape"][idx] = flat_data_dict[param].shape
#
#     if len(dataset_dict["independent_parameters"]["names"]) == 2:
#
#         x_vals = flat_data_dict[dataset_dict["independent_parameters"]["names"][0]]
#         y_vals = flat_data_dict[dataset_dict["independent_parameters"]["names"][1]]
#         # print(dataset_dict['independent_parameters']['shape'][0])
#         datatype = datatype_from_setpoints_2d(x_vals, y_vals)
#
#         if datatype in ("2D_grid", "2D_equidistant"):
#             for idx, param in enumerate(dataset_dict["dependent_parameters"]["names"]):
#
#                 z_vals = (dataset_dict["dependent_parameters"]["data"][idx],)
#                 (
#                     dataset_dict["independent_parameters"]["data"][0],
#                     dataset_dict["independent_parameters"]["data"][1],
#                     dataset_dict["dependent_parameters"]["data"][idx],
#                 ) = reshape_2D_data(x_vals, y_vals, z_vals)
#
#                 dataset_dict["independent_parameters"]["shape"][0] = dataset_dict[
#                     "independent_parameters"
#                 ]["data"][0].shape
#                 dataset_dict["independent_parameters"]["shape"][1] = dataset_dict[
#                     "independent_parameters"
#                 ]["data"][1].shape
#                 dataset_dict["dependent_parameters"]["shape"][idx] = dataset_dict[
#                     "dependent_parameters"
#                 ]["data"][idx].shape
#
#     return dataset_dict
#
#
# def get_data_as_numpy(identifier: Union[int, str], *params: str):
#     """
#     arguments:
#         identifier: guid or run_id
#         *params: dependent (measured) parameters to be included in the returned DataFrame
#                  if no params are included, the default is to return everything
#     returns (1d):
#         x_data (ndarray), [y0_data (ndarray), y1_data (ndarray), ...]
#     returns (2d):
#         x_data (ndarray), y_data (ndarray), [z0_data (ndarray), z1_data (ndarray), ...]
#     """
#
#     dataset = get_dataset_by_identifier(identifier)
#     dataset_dict = _shaped_data_dict(dataset, *params)
#
#     if len(dataset_dict["independent_parameters"]["names"]) == 1:
#         return (
#             dataset_dict["independent_parameters"]["data"][0],
#             dataset_dict["dependent_parameters"]["data"],
#         )
#     elif len(dataset_dict["independent_parameters"]["names"]) == 2:
#         return (
#             dataset_dict["independent_parameters"]["data"][0],
#             dataset_dict["independent_parameters"]["data"][1],
#             dataset_dict["dependent_parameters"]["data"],
#         )
#     else:
#         raise ValueErro(
#             "How did you get this far with 3 independent parameters? Try pandas or xarray"
#         )


def _dataframe_to_xarray(df: DataFrame) -> Dataset:
    """
    Convert pandas DataFrame with MultiIndex to an xarray DataSet.
    """

    df = get_data_as_dataframe(353)
    len_old_df = len(df)
    df = df[~df.index.duplicated()]
    len_new_df = len(df)

    if len_new_df < len_old_df:
        warn("Duplicate values removed from DataFrame. This dataset is weird.")

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
    dependent_parameters = _empty_dataset_dict(dataset)["dependent_parameters"]["names"]
    _check_dependent_params(dependent_parameters, *params)

    dataframe_dict = dataset.get_data_as_pandas_dataframe(*params)

    return pd.concat(list(dataframe_dict.values()), axis=1)


def _check_dependent_params(dependent_parameters: List[str], *params: str):

    for param in params:
        if param not in dependent_parameters:
            raise TypeError(
                f"{param} is not a dependent (measured) parameter of the dataset"
            )
