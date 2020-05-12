from warnings import warn
from qcodes import VisaInstrument
from qcodes import validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping

class Model_7651(VisaInstrument):

    def __init__(self, name, address, **kwargs):

        super().__init__(name, address, terminator="\r\n", **kwargs)

        # self._MODES = {
        #     'dc_volts': 1,
        #     'dc_current': 5,
        # }

        self._VOLT_RANGES = {
            '10mV': 2,
            '100mV': 3,
            '1V': 4,
            '10V': 5,
            '100V': 6,
        }
        self._CURR_RANGES = {
            '1mA': 4,
            '10mA': 5,
            '100mA': 6,
        }
        self._ranges = list(self._VOLT_RANGES.keys()) + list(self._CURR_RANGESa.keys())

        # self.add_parameter(
        #     'mode',
        #     get_cmd=self._get_mode,
        #     set_cmd=self._set_mode,
        #     val_mapping=self._MODES,
        #     docstring='Selects the input mode'
        # )

        self.add_parameter(
            'range',
            get_cmd=self._get_range,
            set_cmd=self._set_range,
            vals=vals.Enum(*self._ranges),
            docstring=('Sets the measurement range.\n'
                      'Note that not only a discrete set of '
                      'ranges can be set (see the manual for '
                      'details).')
        )

        self.add_parameter(
            'output',
            get_cmd=self._get_output,
            set_cmd=self._set_output,
            val_mapping=create_on_off_val_mapping(on_val=1,
                                                  off_val=0)
        )

        self.add_parameter(
            'source',
            set_cmd=self._set_value,
            get_cmd=self._get_value,
        )

    def read(self):
        return self.visa_handle.read()

    def _get_panel_setting(self):
        # this is goofy as hell
        self.write('OS')
        output = []
        while True:
            line = self.read()
            if line=='END':
                break
            output.append(line)
        return output

    def _set_source_units(self, mode):
        if mode==1:
            self.source.unit = 'V'
        elif mode==5:
            self.source.unit = 'A'

    def _set_mode(self, mode):
        self.write(f'F{mode:d}')
        self.visa_handle.assert_trigger()
        self._set_source_units(mode)

    def _get_mode(self):
        os_output = self._get_panel_setting()
        mode = int(os_output[1][1])
        self._set_source_units(mode)
        return mode

    def _set_range(self, range):
        if 'V' in range:
            mode = 1
            setting = self._VOLT_RANGES[range]
        elif 'A' in range:
            mode = 5
            setting = self._CURR_RANGES[range]
        self._set_mode(mode)
        self.write(f'R{setting:d}')
        self.visa_handle.assert_trigger()

    def _get_range(self):
        os_output = self._get_panel_setting()
        mode = int(os_output[1][1])
        range = int(os_output[1][3])
        if mode==1:
            range = next(key for key, val in self._VOLT_RANGES.items() if val==range)
        elif mode==5:
            range = next(key for key, val in self._CURR_RANGES.items() if val==range)
        return range

    def _set_output(self, output):
        self.write(f'O{output:d}')
        self.visa_handle.assert_trigger()

    def _get_output(self):
        sts = self.ask('OC')
        sts = int(sts.split('=')[1])
        sts_bits = '{0:08b}'.format(sts)
        return int(sts_bits[-5])

    def _set_value(self, val):
        command = 'S{:+0.2E}'.format(val)
        self.write(command)
        self.visa_handle.assert_trigger()

    def _get_value(self):
        output = self.ask('OD')
        overload = output[0]
        if overload=='E':
            warn('Source is overloaded!')
        unit = output[3]
        self.source.unit = unit
        return float(output[4:])
