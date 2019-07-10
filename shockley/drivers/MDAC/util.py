''' helper functions for the MDAC '''

# kill this MDAC package and move this to sqpurdue along with Thorvald's driver

import time
import numpy as np

# convert chip carrier pins to mdac channel numbers
# note: mdac channel index = mdac channel number - 1
LCC_TO_MDACNUM = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8,
    9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
    17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24,
    25:26, 26:27, 27:28, 28:29, 29:30, 30:31, 31:32, 32:33,
}

def setup_default_limits(md, channel_list=None):
    ''' channel list should be a channel number not an index '''
    md.protection('on')

    if channel_list is None:
        channel_list = list(range(64))
    else:
        channel_list = [i-1 for i in channel_list]

    for i in channel_list:
        ch = md.channels[i]
        try:
            ch._set_limits(minlimit=-5.0, maxlimit=5.0, ratelimit=1000.0)
        except Exception as e:
            # don't try to do anything smart
            # just print out any problems
            print(ch.channel_number(), e)

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


def microd_1M_ground(md, channel_list=None):
    ''' open all microD relays, which connects them to ground with 1M

        channel_list uses mdac numbering, not index'''

    if channel_list is None:
        channel_list = list(range(64))
    else:
        channel_list = [i-1 for i in channel_list]

    for i in channel_list:
        ch = md.channels[i]
        try:
            ch.microd.set('open')
        except Exception as e:
            # these will have to be fixed manually
            print(ch.channel_number(), e)


def all_microd_direct_ground(md, channel_list):
    ''' ground all microD pins directly

        channel_list uses mdac numbering, not index'''

    if channel_list is None:
        channel_list = list(range(64))
    else:
        channel_list = [i-1 for i in channel_list]

    for i in channel_list:
        ch = md.channels[i]
        try:
            ch.microd_ground()
        except Exception as e:
            # these will have to be fixed manually
            print(ch.channel_number(), e)
