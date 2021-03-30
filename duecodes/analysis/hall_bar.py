from typing import Optional, List, Union, Tuple
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
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

def clean_hall_data(
    identifier: Union[int, str],
    vxx_param_name: str, vxy_param_name: str,
    bias_current: float, width: float, length: float,
    voltage_amp_xx: float = 1.0, voltage_amp_xy: float = 1.0
) -> xr.Dataset:

    ds = get_dataset_by_identifier(identifier)
    field_param, gate_param = _get_field_gate_params(ds)

    arr = ds.to_xarray_dataset()
    arr = arr.rename({gate_param: 'gate', field_param: 'field'})


    arr[vxx_param_name] = arr[vxx_param_name].real/voltage_amp_xx # vxx
    arr[vxy_param_name] = arr[vxy_param_name].real/voltage_amp_xy # vxy

    arr[vxx_param_name] = \
        (arr[vxx_param_name]/bias_current)/(length/width) # rxx, Ohms per square
    arr[vxy_param_name] = (arr[vxy_param_name]/bias_current) # rxy, Ohms

    return arr.rename({vxx_param_name: 'R_xx', vxy_param_name: 'R_xy'})

def calculate_density_mobility(data: xr.Dataset) -> xr.Dataset:

    rxx = data['R_xx']
    rxy = data['R_xy']

    popt = rxy.polyfit('field', 1).drop('degree')

    results = xr.full_like(
        rxx[0,:], np.nan
    ).drop('field').rename('density').to_dataset()
    results['mobility'] = xr.full_like(results['density'], np.nan).rename('mobility')

    results['density'] = (1/(popt['polyfit_coefficients'][0]*ELEC_CHARGE)*(1e-4)) # 1/cm^2
    results['mobility'] = 1e4*(popt['polyfit_coefficients'][0])/rxx.loc[0,:]

    return results

def plot_density_mobility(
    results: xr.Dataset
    ) -> Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]:

    fig, ax = plt.subplots(1,1)

    im = ax.scatter(results['density'], results['mobility'],
        c=results['gate'])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('V_gate (V)')

    v_gate_at_max = results['mobility'].idxmax()
    u_max = results['mobility'].loc[v_gate_at_max].values
    n_at_max = results['density'].loc[v_gate_at_max].values
    print(f'{u_max:.2e}, {n_at_max:.2e}')

    at = AnchoredText(f'{u_max:.2e} cm^2/(Vs)\nat {n_at_max:.2e} 1/cm^2',
                      loc='upper right', prop=dict(size=10), frameon=True,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    ax.set_ylabel('mobility (cm^2/(Vs))')
    ax.set_xlabel('density (1/cm^2)')
    ax.grid(ls=':')

    fig.tight_layout()

    return ax, cb
