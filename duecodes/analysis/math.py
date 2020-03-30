import numpy as np


def xy_to_meshgrid(xrow, yrow):
    # stolen from qcodes.dataset.plotting
    # we use a general edge calculator,
    # in the case of non-equidistantly spaced data
    # TODO: is this appropriate for a log ax?

    dxs = np.diff(xrow) / 2
    dys = np.diff(yrow) / 2

    x_edges = np.concatenate(
        (np.array([xrow[0] - dxs[0]]), xrow[:-1] + dxs, np.array([xrow[-1] + dxs[-1]]))
    )
    y_edges = np.concatenate(
        (np.array([yrow[0] - dys[0]]), yrow[:-1] + dys, np.array([yrow[-1] + dys[-1]]))
    )

    return np.meshgrid(x_edges, y_edges)


def moving_avg(x, y, avgs, axis=None):

    xx = np.cumsum(x, dtype=np.float)
    xx[avgs:] = xx[avgs:] - xx[:-avgs]
    xx = xx[avgs - 1 :] / avgs

    if axis == 0:
        ret = np.cumsum(y, axis=0, dtype=np.float)
        ret[avgs:] = ret[avgs:] - ret[:-avgs]
        return xx, ret[avgs - 1 :] / avgs
    elif axis == 1:
        ret = np.cumsum(y, axis=1, dtype=np.float)
        ret[:, avgs:] = ret[:, avgs:] - ret[:, :-avgs]
        return xx, ret[:, avgs - 1 :] / avgs
    else:
        ret = np.cumsum(y, dtype=np.float)
        ret[avgs:] = ret[avgs:] - ret[:-avgs]
        return xx, ret[avgs - 1 :] / avgs
