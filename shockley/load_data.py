from typing import Optional, Collection, Union
from qcodes import load_by_id, load_by_guid
# from qcodes.dataset.plotting import plot_by_id
# from qcodes.dataset.data_export import get_data_by_id, get_1D_plottype, get_2D_plottype, \
#                                        flatten_1D_data_for_plot, reshape_2D_data
# from collections import OrderedDict
#
# def get_param_dict(ds):
#     ''' returns {'name':ParamSpec, ....} with all parameters in the dataset '''
#     out = {}
#     for i, param in enumerate(ds.get_parameters()):
#         out[param.name] = param
#     return out
#
# def get_structure(ds):
#     structure = OrderedDict({})
#     param_dict = get_param_dict(ds)
#
#     # for each data param (non-independent param)
#     for dependent in ds.dependent_parameters:
#
#         # get name etc.
#         name = dependent.name
#         structure[name] = {'unit': dependent.unit}
#         structure[name]['dependencies'] = []
#
#         # find dependencies (i.e., axes) and
#         # add their names/units in the right order
#         dependencies = param_dict[name].depends_on_
#         for iax, dep in enumerate(dependencies):
#             dep_param = param_dict[dep]
#             dep_struct = {'name': dep_param.name,
#                           'unit': dep_param.unit}
#             structure[name]['dependencies'].insert(iax, dep_struct)
#
#     return structure
#
#
# def get_axes_from_structure(struct, param_name):
#     axes_names = []
#     axes_vals = []
#
#     if type(param_name) == list:
#         param_name = param_name[0]
#
#     axes_info = struct[param_name]['dependencies']
#     for ax in axes_info:
#         n = "{}".format(ax['name'])
#         if ax['unit'] != '':
#             n += " ({})".format(ax['unit'])
#
#         axes_names.append(n)
#
#     return axes_names

def get_dataset_by_identifier(
    identifier: Union[int, str],
):
    '''
        returns a dataset for a given identifier

        identifier: run_id or guid
    '''
    pass


def list_measured_params(
    identifier: Union[int, str],
    print_output: bool = False
):
    pass
    # ds = load_by_id(dat)
    # struct = get_structure(ds)
    # out = {}
    # for k in struct:
    #     out[k] = get_axes_from_structure(struct, k)
    # if print_:
    #     print(json.dumps(out, sort_keys=True, indent=2))
    # return out


# def xy_to_meshgrid(xrow, yrow):
#     # stolen from qcodes.dataset.plotting
#     # we use a general edge calculator,
#     # in the case of non-equidistantly spaced data
#     # TODO: is this appropriate for a log ax?
#
#     dxs = np.diff(xrow)/2
#     dys = np.diff(yrow)/2
#
#     x_edges = np.concatenate((np.array([xrow[0] - dxs[0]]),
#                               xrow[:-1] + dxs,
#                               np.array([xrow[-1] + dxs[-1]])))
#     y_edges = np.concatenate((np.array([yrow[0] - dys[0]]),
#                               yrow[:-1] + dys,
#                               np.array([yrow[-1] + dys[-1]])))
#
#     return np.meshgrid(x_edges, y_edges)

# def data_on_a_plain_grid(x, y, z):
#     ''' returns reshaped x, y, and z data for gridded data sets'''
#
#     x_is_stringy = isinstance(x[0], str)
#     y_is_stringy = isinstance(y[0], str)
#     z_is_stringy = isinstance(z[0], str)
#
#     if x_is_stringy:
#         x_strings = np.unique(x)
#         x = _strings_as_ints(x)
#
#     if y_is_stringy:
#         y_strings = np.unique(y)
#         y = _strings_as_ints(y)
#
#     if z_is_stringy:
#         z_strings = np.unique(z)
#         z = _strings_as_ints(z)
#
#     return reshape_2D_data(x, y, z)


# def get_data_as_xarray(identifier, param_names=None):
#     ''' return parameter data from dataset as a dictionary of xarrays
#         whatever the hell that means.
#
#         param_names: list of parameters, if None, all parameters are returned
#      '''
#
#     if param_name in list_measured_params(dat):
#         pass
#     else:
#         raise ValueError(f'{param_name} is not a measured parameter')
#
#     ds = load_by_id(dat)
#     df = ds.get_data_as_pandas_dataframe()[param_name]
#
#     if dtype=='pandas':
#         return df
#
#     elif dtype=='numpy':
#         # logic in here is shamelessly stolen from plot_by_id
#
#         ax_names = df.index.names
#
#         if len(ax_names) == 1:  # 1D
#
#             xname = ax_names[0]
#             xpoints = flatten_1D_data_for_plot(df.index.values)
#
#             yname = param_name
#             ypoints = flatten_1D_data_for_plot(df.values)
#
#             plottype = get_1D_plottype(xpoints, ypoints)
#
#             if plottype in ['1D_line', '1D_point', '1D_bar']:
#                 return {'y': {'name':yname, 'vals': ypoints}, 'x': {'name':xname, 'vals': xpoints}}
#
#             else:
#                 raise ValueError('Unknown data shape. Something is way wrong.')
#
#
#         elif len(ax_names) == 2:  # 2D
#
#             # From the setpoints, figure out what sort of 2D data this is
#
#             xname = ax_names[0]
#             xpoints = flatten_1D_data_for_plot(df.index.get_level_values(0))
#
#             yname = ax_names[1]
#             ypoints = flatten_1D_data_for_plot(df.index.get_level_values(1))
#
#             zname = param_name
#             zpoints = flatten_1D_data_for_plot(df.values)
#
#             plottype = get_2D_plottype(xpoints, ypoints, zpoints)
#
#             if (plottype=='2D_grid' or plottype=='2D_equidistant'):
#                 x, y, z = reshape_2D_data(xpoints, ypoints, zpoints)
#                 return {'z': {'name':zname, 'vals':z},
#                         'x': {'name':xname, 'vals': x}, 'y': {'name':yname, 'vals': y}}
#
#             elif (plottype=='2D_point'or plottype=='2D_unknown'):
#                 return {'z': {'name':zname, 'vals':zpoints},
#                         'x': {'name':xname, 'vals': xpoints}, 'y': {'name':yname, 'vals': ypoints}}
#
#             else:
#                 raise ValueError('Unknown data shape. Something is way wrong.')
#
#         else:
#             raise ValueError('MultiDimensional data encountered.')
#
#     else:
#         raise ValueError('dtype should be `numpy` or `pandas`')
