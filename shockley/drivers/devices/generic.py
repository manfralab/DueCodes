import json
from datetime import datetime
import numbers
import numpy as np

from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.helpers import NumpyJSONEncoder, full_class
from qcodes.utils.validators import Ints, Bool

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

### analysis ###

COND_QUANT =  7.748091729e-5 # Siemens

def _dfdx(f, x, axis = None):
    # returns df(x)/dx
    dx = (x - np.roll(x,1))[1:].mean()
    return np.gradient(f,dx, axis = axis)

### dummy contact/gate ###

class DummyChan(InstrumentChannel):

    def __init__(self, parent, name, chip_num):

        """
        Args:
            parent (Instrument): The device the result is extract from
            name (str): the name of the extracted result
            chip_num (int): pin number on chip carrier/socket/daughter board
        """

        self._dev = parent
        self._chip_num = chip_num
        self._mdac_channel = self._dev._pin_map[chip_num]

        super().__init__(parent, name)
        self.name = name

        self.add_parameter('failed',
                           label='Contact Failed',
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=False,
                           vals=Bool())

    def chip_number(self):
        """ number on the chip carrier"""
        return self._chip_num

    def mdac_number(self):
        """ DAC channel number corresponding to SMC channel numbers"""
        return self._mdac_channel

    def _to_dict(self):

        snap = {
            "name": self.name,
            "chip_number": self.chip_number(),
            "mdac_number": self.mdac_number(),
            "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "failed": self.failed(),
            }

        return snap

### channels ###

class Result(InstrumentChannel):

    def __init__(self, parent, name, unit):

        """
        Args:
            parent (Instrument): The device the result is extract from
            name (str): the name of the extracted result
            units (str): units of the extracted result
        """

        super().__init__(parent, name)
        self.name = name

        self.add_parameter('run_id',
                           label='Source Data Run Id',
                           get_cmd=None,
                           set_cmd=None,
                           vals=Ints())

        self.add_parameter('val',
                           label=f'{name} ({unit})',
                           get_cmd=None,
                           set_cmd=None,
                           unit=unit)

    def _to_dict(self):

        snap = {
            "name": self.name,
            "val": self.val(),
            "run_id": self.run_id(),
            "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)

        return snap

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

        self.add_parameter('failed',
                           label='Contact Failed',
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=False,
                           vals=Bool())

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

    def _to_dict(self):

        snap = {
            "name": self.name,
            "chip_number": self.chip_number(),
            "mdac_number": self.mdac_number(),
            "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "failed": self.failed(),
            }

        snap["voltage"] = self.voltage.snapshot(),
        snap["relays:"] = {
                            "dac_output": self.dac_output(),
                            "bus": self.bus(),
                            "gnd": self.gnd(),
                            "microd": self.microd(),
                            "smc": self.smc(),
                            }
        return snap


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

        self.add_parameter('failed',
                           label='Gate Failed',
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=False,
                           vals=Bool())

    def chip_number(self):
        """ number on the chip carrier"""
        return self._chip_num

    def mdac_number(self):
        """ DAC channel number corresponding to SMC channel numbers"""
        return self._mdac_channel

    def _to_dict(self):

        snap = {
            "name": self.name,
            "chip_number": self.chip_number(),
            "mdac_number": self.mdac_number(),
            "ts": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "failed": self.failed(),
            }

        snap["voltage"] = self.voltage.snapshot(),
        snap["relays:"] = {
                            "dac_output": self.dac_output(),
                            "bus": self.bus(),
                            "gnd": self.gnd(),
                            "microd": self.microd(),
                            "smc": self.smc(),
                           }
        return snap

### load/save device classes ###

def _mdac_simple_dict(obj):

        snap = {
            "__class__": full_class(obj)
        }

        for attr in set(obj._meta_attrs):
            if hasattr(obj, attr):
                snap[attr] = getattr(obj, attr)

        return snap

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

            # special case for MDAC, because that mess takes too long...
            if 'MDAC' in str(obj):
                return _mdac_simple_dict(obj)
            else:
                return obj.snapshot(update=False)

        elif isinstance(obj, (Result, Contact, Gate)):
            return obj._to_dict()

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

### measurements ###

def _meas_gate_leak(v_gate_param, i_param, v_step, v_max, delay,
                    di_limit=None, compliance=None, plot_logs=False, write_period=0.1):

    meas = Measurement()
    meas.write_period = write_period

    meas.register_parameter(v_gate_param)
    v_gate_param.post_delay = 0

    meas.register_parameter(i_param, setpoints=(v_gate_param,))

    with meas.run() as ds:

        plot_subscriber = QCSubscriber(ds.dataset, v_gate_param, [i_param],
                                       grid=None, log=plot_logs)
        ds.dataset.subscribe(plot_subscriber)

        curr = np.array([])
        for vg in gen_sweep_array(0, v_max, step=v_step):

            v_gate_param.set(vg)
            time.sleep(delay)

            curr = np.append(curr, i_param.get())
            ds.add_result((v_gate_param, vg),
                          (i_param, curr[-1]))

            if vg>0:
                if np.abs(curr.max() - curr.min()) > di_limit:
                    print('Leakage current limit exceeded!')
                    vmax = vg-v_step # previous step was the limit
                    break
                elif np.abs(curr[-1]) > compliance:
                    print('Current compliance level exceeded!')
                    vmax = vg - v_step # previous step was the limit
                    break
        else:
            vmax = v_max

        for vg in gen_sweep_array(vmax, 0, step=volt_step):

            v_gate_param.set(vg)
            time.sleep(delay)

            curr = np.append(curr, i_param.get())
            ds.add_result((v_gate_param, vg),
                          (i_param, curr[-1]))

        return ds.run_id, vmax
