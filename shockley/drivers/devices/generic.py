import json
import numbers
import numpy as np

from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.helpers import NumpyJSONEncoder

from shockley.drivers.MDAC.extMDAC import MDACChannel

### socket maps ###

LCC_MAP = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8,
    9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
    17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24,
    25:26, 26:27, 27:28, 28:29, 29:30, 30:31, 31:32, 32:33,
}

DIRECT_MAP = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8,
    9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
    17:17, 18:18, 19:19, 20:20, 21:21, 22:22, 23:23, 24:24,
    25:25, 26:26, 27:27, 28:28, 29:29, 30:30, 31:31, 32:32,
    33:33, 34:34, 35:35, 36:36, 37:37, 38:38, 39:39, 40:40,
    41:41, 42:42, 43:43, 44:44, 45:45, 46:46, 47:47, 48:48,
}

### channels ###

class Contact(MDACChannel):
    ''' class for ohmic contacts '''

    def __init__(self, parent, name, chip_num):
        """
        Args:
            parent (Device): The device the channel is a part of.
            name (str): the name of the channel
            chip_num (int): pin number for this contact according to the fridge
        """

        self._dev = parent
        self._chip_num = chip_num
        self._mdac_channel = self._dev._pin_map[chip_num]

        super().__init__(self._dev._mdac, name, self._mdac_channel)
        self.name = name

    def chip_number(self):
        """ number on the chip carrier"""
        return self._chip_num

    def mdac_number(self):
        """ DAC channel number corresponding to SMC channel numbers"""
        return self._mdac_channel

    def ac_output_on(self, frequency, amplitude, offset=0.0):

        amplitude = np.abs(amplitude) # just to be sure

        # check rate
        max_rate = 2*np.pi*frequency*amplitude
        if max_rate > self.limit_rate.get():
            raise ValueError(f'limit_rate for MDAC channel too slow for this frequency output. Need {max_rate:.2f} V/s or greater')

        # check amplitudes
        vp = np.sqrt(2)*amplitude # peak voltage
        vmin = min(-1*vp - offset, -1*vp + offset)
        vmax = max(vp - offset, vp + offset)
        if vmax>self.limit_max.get():
            raise ValueError(f'limit_max for MDAC channel is too low for this amplitude/offset. Need{vmax:.2f} V or greater')
        elif vmin<self.limit_min.get():
            raise ValueError(f'limit_min for MDAC channel is too high for this amplitude/offset. Need{vmin:.2f} V or less')

        # start output
        self.awg_sine(frequency, np.sqrt(8)*amplitude, offset=offset, phase=0)
        self.attach_trigger()
        self._parent.trigger0.direction('up')
        self._parent.trigger0.start(frequency)
        self._parent.sync()

    def ac_output_off(self, offset=0.0):

        # turn off AC bias
        self._parent.trigger0.stop()
        self.awg_off()
        self.voltage.set(offset)

class Gate(MDACChannel):
    ''' class for gates '''

    def __init__(self, parent, name, chip_num):
        """
        Args:
            parent (Device): The device the channel is a part of.
            name (str): the name of the channel
            chip_num (int): pin number for this contact according to the fridge
        """

        self._dev = parent
        self._chip_num = chip_num
        self._mdac_channel = self._dev._pin_map[chip_num]

        super().__init__(self._dev._mdac, name, self._mdac_channel)
        self.name = name

    def chip_number(self):
        """ number on the chip carrier"""
        return self._chip_num

    def mdac_number(self):
        """ DAC channel number corresponding to SMC channel numbers"""
        return self._mdac_channel

### functions ###

def _dfdx(f, x, axis = None):
    # returns df(x)/dx
    dx = (x - np.roll(x,1))[1:].mean()
    return np.gradient(f,dx, axis = axis)

### load/save ###

class devJSONEncoder(json.JSONEncoder):
    """
    This JSON encoder adds support for serializing types that the built-in
    `json` module does not support out-of-the-box. See the docstring of the
    `default` method for the description of all conversions.
    """

    def default(self, obj):
        """
        List of conversions that this encoder performs:
        * `numpy.generic` (all integer, floating, and other types) gets
        converted to its python equivalent using its `item` method (see
        `numpy` docs for more information,
        https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html)
        * `numpy.ndarray` gets converted to python list using its `tolist`
        method
        * complex number (a number that conforms to `numbers.Complex` ABC) gets
        converted to a dictionary with fields "re" and "im" containing floating
        numbers for the real and imaginary parts respectively, and a field
        "__dtype__" containing value "complex"
        * qcodes.
        * object with a `_JSONEncoder` method get converted the return value of
        that method
        * objects which support the pickle protocol get converted using the
        data provided by that protocol
        * other objects which cannot be serialized get converted to their
        string representation (suing the `str` function)
        """

        # numpy
        if isinstance(obj, np.generic) \
                and not isinstance(obj, np.complexfloating):
            # for numpy scalars
            return obj.item()
        elif isinstance(obj, np.ndarray):
            # for numpy arrays
            return obj.tolist()
        elif (isinstance(obj, numbers.Complex) and
              not isinstance(obj, numbers.Real)):
            return {
                '__dtype__': 'complex',
                're': float(obj.real),
                'im': float(obj.imag)
            }

        # qcodes paramters/instruments
        elif isinstance(obj, (Instrument, Parameter)):
            return obj.snapshot(update=False)

        # other/fallback
        elif hasattr(obj, '_JSONEncoder'):
            # Use object's custom JSON encoder
            return obj._JSONEncoder()
        else:
            try:
                s = json.JSONEncoder.default(self, obj)
            except TypeError:
                # See if the object supports the pickle protocol.
                # If so, we should be able to use that to serialize.
                if hasattr(obj, '__getnewargs__'):
                    return {
                        '__class__': type(obj).__name__,
                        '__args__': obj.__getnewargs__()
                    }
                else:
                    # we cannot convert the object to JSON, just take a string
                    s = str(obj)
            return s
