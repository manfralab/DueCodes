import numpy as np

def moving_avg(x, y, avgs, axis = None) :

    xx = np.cumsum(x, dtype=np.float)
    xx[avgs:] = xx[avgs:] - xx[:-avgs]
    xx = xx[avgs - 1:] / avgs

    if axis==0:
        ret = np.cumsum(y, axis=0, dtype=np.float)
        ret[avgs:] = ret[avgs:] - ret[:-avgs]
        return xx, ret[avgs - 1:] / avgs
    elif axis==1:
        ret = np.cumsum(y, axis=1, dtype=np.float)
        ret[:,avgs:] = ret[:,avgs:] - ret[:,:-avgs]
        return xx, ret[:,avgs - 1:] / avgs
    else:
        ret = np.cumsum(y, dtype=np.float)
        ret[avgs:] = ret[avgs:] - ret[:-avgs]
        return xx, ret[avgs - 1:] / avgs
