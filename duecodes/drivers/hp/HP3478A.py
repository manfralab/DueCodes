from qcodes import VisaInstrument
from qcodes import validators as vals

class HP3478A(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\r\n", **kwargs)

        self._MODES = {
            'dc_volts': 1,
            'ac_volts': 2,
            '2_wire_res': 3,
            '4_wire_res': 4,
            'dc_current': 5,
            'ac_current': 6,
            'extended_res': 7,
        }

        self._RANGES = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 'A']

        self._NPLC = {
            0.1: 1,
            1: 2,
            10: 3,
        }

        self.add_parameter(
            'mode',
            # get_cmd=None,
            set_cmd=self._set_mode,
            val_mapping=self._MODES,
            docstring='Selects the input mode'
        )

        self.add_parameter(
            'range',
            # get_cmd=None,
            set_cmd=self._set_range,
            vals=vals.Enum(*self._RANGES),
            docstring=('Sets the measurement range.\n'
                      'Note that not only a discrete set of '
                      'ranges can be set (see the manual for '
                      'details).')
        )

        self.add_parameter(
            'nplc',
            # get_cmd=None,
            set_cmd='N{:d}',
            val_mapping = self._NPLC,
            unit='APER',
            docstring=('Get integration time in Number of '
                      'PowerLine Cycles.\n'
                      'To get the integrationtime in seconds, '
                      'use get_integrationtime().')
            )

        self.add_parameter(
            'fetch',
            get_cmd=self._get_reading,
            docstring=('Get most recent reading. Try to get the units right.')
        )

    def _set_mode(self, mode):

        if mode in [1, 2]:
            self.fetch.unit = 'V'
        elif mode in [3, 4, 7]:
            self.fetch.unit = 'Ohm'
        elif mode in [5, 6]:
            self.fetch.unit = 'A'

        self.write(f'F{mode:d}')

    def _set_range(self, range):
        if isinstance(range, int):
            self.write(f'R{range:d}')
        elif isinstance(range, str):
            range = range.strip()
            self.write(f'R{range}')

    def _get_reading(self):
        rdng = self.visa_handle.read()
        return float(rdng)
