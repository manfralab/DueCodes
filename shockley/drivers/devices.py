''' trying to come up with a reasonable way to address a device
    connected through an MDAC.

    currently just works for a single MDAC '''

from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.base import Instrument
from .MDAC.extMDAC import MDACChannel

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

class Contact(MDACChannel):
    ''' class for ohmic contacts '''

    def __init__(self, parent, name, chip_num):
        """
        Args:
            parent (Device): The device the channel is a part of.
            name (str): the name of the channel
            chip_num (int): pin number for this contact according to the fridge
        """

        self._parent = parent
        self._chip_num = chip_num
        self._mdac_channel = self._parent._pin_map[chip_num]

        super().__init__(self._parent._mdac, name, self._mdac_channel)

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

        self._parent = parent
        self._chip_num = chip_num
        self._mdac_channel = self._parent._pin_map[chip_num]

        super().__init__(self._parent._mdac, name, self._mdac_channel)

    def chip_number(self):
        """ number on the chip carrier"""
        return self._chip_num

    def mdac_number(self):
        """ DAC channel number corresponding to SMC channel numbers"""
        return self._mdac_channel


class FET(Instrument):
    ''' full device class '''

    def __init__(self, name, md, sources=None, drain=None, gate=None, chip_carrier=None, **kwargs):
        '''
        Args:
            name (str): the name of the device
            contact_num (list of ints): list of pin numbers for contacts to the device
            gate_num (list of ints): list of pin numbers for gates on the device
            parent (MDAC): MDAC that the device is connected to
            **kwargs are passed to qcodes.instrument.base.Instrument
        '''

        self._mdac = md

        if chip_carrier is None:
            self._pin_map = DIRECT_MAP
        elif chip_carrier=='lcc':
            self._pin_map = LCC_MAP
        else:
            raise ValueError('pin to MDAC mapping not defined')

        super().__init__(name, **kwargs)

        ### check some inputs ###
        if sources is None:
            # this is only a kwarg for readability
            raise ValueError('define a list of source contacts')

        if drain is None:
            # this is only a kwarg for readability
            raise ValueError('define a drain contact')

        if gate is None:
            # this is only a kwarg for readability
            raise ValueError('define a gate contact')

        ### create source submodules
        allsource = ChannelList(self, "Sources", Contact,
                                snapshotable=False)

        for s in sources:
            source = Contact(self, f'{name}_s{s:02}', s)
            allsource.append(source)
            self.add_submodule('s{:02}'.format(s), source)

        allsource.lock()
        self.add_submodule('sources', allsource)

        ### create gate submodule
        g = Gate(self, f'{name}_gate', gate)
        self.add_submodule('gate', g)

        ### create drain submodule
        d = Contact(self, f'{name}_drain', drain)
        self.add_submodule(f'drain', d)
