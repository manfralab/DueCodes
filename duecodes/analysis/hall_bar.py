from typing import Optional, List, Union, Tuple
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from qcodes.dataset.data_set import DataSet
from duecodes.analysis.load import get_dataset_by_identifier

ELEC_CHARGE = 1.602e-19 # C

def _get_field_gate_params(ds: DataSet) -> Tuple[str, str]:
    field_param = ''
    gate_param = ''

    for spec in ds.paramspecs.values():
        if not spec.depends_on:
            if 'T' in spec.unit:
                field_param = spec.name
            elif 'V' in spec.unit:
                gate_param = spec.name
    if (not field_param) or (not gate_param):
        raise KeyError('Could not find magnetic field or gate voltage parameter.')
    return field_param, gate_param


def calculate_density_mobility(
    identifier: Union[int, str],
    vxx_param_name: str, vxy_param_name: str,
    bias_current: float, width: float, length: float,
    voltage_amp_xx: float = 1.0, voltage_amp_xy: float = 1.0
) -> xr.Dataset:

    ds = get_dataset_by_identifier(identifier)
    field_param, gate_param = _get_field_gate_params(ds)

    arr = ds.to_xarray_dataset()
    arr = arr.rename({gate_param: 'gate', field_param: 'field'})

    vxx = arr[vxx_param_name].real/voltage_amp_xx
    vxy = arr[vxy_param_name].real/voltage_amp_xy

    rxx = (vxx/bias_current).rename('R_xx')/(length/width) # R per square
    rxy = (vxy/bias_current).rename('R_xy')

    popt = rxy.polyfit('field', 1).drop('degree')

    results = xr.full_like(
        rxx[0,:], np.nan
    ).drop('field').rename('density').to_dataset()
    results['mobility'] = xr.full_like(results['density'], np.nan).rename('mobility')

    results['density'] = (1/(popt['polyfit_coefficients'][0]*ELEC_CHARGE)*(1e-4)) # 1/cm^2
    results['mobility'] = 1e4*(popt['polyfit_coefficients'][0])/rxx.loc[0,:]

    return results

def plot_density_mobility(
    arr: xr.Dataset
    ) -> Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]:

    fig, ax = plt.subplots(1,1)

    im = ax.scatter(arr['density'], arr['mobility'], c=arr['gate'])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('V_gate (V)')

    ax.set_ylabel('mobility (cm^2/(Vs))')
    ax.set_xlabel('density (1/cm^2)')
    ax.grid(ls=':')

    fig.tight_layout()

    return ax, cb
