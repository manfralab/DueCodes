from typing import Optional, List, Union
import pprint
from warnings import warn
import numpy as np
import pandas as pd
from pandas import DataFrame
from xarray import Dataset
from qcodes.dataset.data_set import load_by_run_spec, load_by_guid
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

def list_parameters(identifier: Union[int, str]):
    ds = get_dataset_by_identifier(identifier)
    output = {
        'dependent': [],
        'independent': [],
        }

    for pspec in ds.paramspecs.values():
        if not pspec.depends_on:
            output['independent'].append(pspec.name)
        else:
            output['dependent'].append(pspec.name)

    pp = pprint.PrettyPrinter(indent=2, sort_dicts=False)
    pp.pprint(output)
