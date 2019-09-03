''' helper functions for the MDAC '''

import time
import numpy as np

def ac_output_on(channel, frequency, amplitude, offset=0.0):
    # setup ac output on MDAC channel with trigger0

    amplitude = np.abs(amplitude) # just to be sure

    # check rate
    max_rate = 2*np.pi*frequency*amplitude
    if max_rate > channel.limit_rate.get():
        raise ValueError(f'limit_rate for MDAC channel too slow for this frequency output. Need {max_rate:.2f} V/s or greater')

    # check amplitudes
    vp = np.sqrt(2)*amplitude # peak voltage
    vmin = min(vp - offset, vp + offset)
    vmax = max(vp - offset, vp + offset)
    if vmax>channel.limit_max.get():
        raise ValueError(f'limit_max for MDAC channel is too low for this amplitude/offset. Need{vmax:.2f} V or greater')
    elif vmin<channel.limit_min.get():
        raise ValueError(f'limit_min for MDAC channel is too high for this amplitude/offset. Need{vmin:.2f} V or less')

    # start output
    channel.awg_sine(frequency, np.sqrt(8)*amplitude, offset=offset, phase=0)
    channel.attach_trigger()
    channel._parent.trigger0.direction('up')
    channel._parent.trigger0.start(frequency)
    channel._parent.sync()

def ac_output_off(channel):

    # turn off AC bias
    channel._parent.trigger0.stop()
    channel.awg_off()
    channel.voltage.set(0.0)
