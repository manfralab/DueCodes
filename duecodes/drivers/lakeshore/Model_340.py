''' Minimal working driver for Lakeshore Model 340 Temp Controller '''

from typing import Dict, ClassVar, Any, Optional
from qcodes import InstrumentChannel, VisaInstrument, ChannelList
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import (
    LakeshoreBase,
    BaseOutput,
    BaseSensorChannel,
)
from qcodes.instrument.group_parameter import GroupParameter, Group
import qcodes.utils.validators as vals


class Output_340(InstrumentChannel):
    """Class for control outputs (heaters) of model 340"""

    RANGES: ClassVar[Dict[str, int]] = {
        "off": 0,
        "625uW": 1,
        "6.2mW": 2,
        "62mW": 3,
        "625mW": 4,
        "6.2W": 5,
    }

    def __init__(
            self,
            parent: "Model_340",
            output_name: str,
            loop: Optional[int] = None,
            has_pid: bool = True):

        super().__init__(parent, output_name)

        self._loop = loop

        self.add_parameter(
            "range",
            label="heater range",
            docstring="set heater output range",
            val_mapping=self.RANGES,
            set_cmd=f'RANGE {{}}',
            get_cmd=f'RANGE?'
        )

        if self._loop:

            self._loop = loop

            self.add_parameter(
                "setpoint",
                label="setpoint temp",
                docstring="setpoint for heater control loop",
                vals=vals.Numbers(0, 400),
                get_parser=float,
                set_cmd=f'SETP {self._loop}, {{}}',
                get_cmd=f'SETP? {self._loop}'
            )

            self.add_parameter('input_channel',
                               label='Input channel',
                               docstring='Specifies which measurement input to '
                                         'control from (note that only '
                                         'measurement inputs are available)',
                               parameter_class=GroupParameter,
                               get_parser=str,)

            self.add_parameter('units',
                               label='Specifies setpoint units',
                               docstring='Specifies whether the output remains on '
                                         'or shuts off after power cycle.',
                               val_mapping={'Kelvin': 1, 'Celsius': 2, 'Sensor': 3},
                               parameter_class=GroupParameter)

            self.add_parameter('control_loop_status',
                               label='Power-up enable on/off',
                               docstring='Specifies whether the output remains on '
                                         'or shuts off after power cycle.',
                               val_mapping={'on':1, 'off':0},
                               parameter_class=GroupParameter)

            self.add_parameter('powerup_enable',
                               label='Power-up enable on/off',
                               docstring='Specifies whether the output remains on '
                                         'or shuts off after power cycle.',
                               val_mapping={True: 1, False: 0},
                               parameter_class=GroupParameter)

            self.output_group = Group([self.input_channel, self.units,
                                       self.control_loop_status, self.powerup_enable],
                                      set_cmd=f'CSET {self._loop}, '
                                              f'{{input_channel}}, '
                                              f'{{units}}, '
                                              f'{{control_loop_status}}, '
                                              f'{{powerup_enable}}',
                                      get_cmd=f'CSET? {loop}')

        if 'heater' in output_name:

            self.add_parameter('output',
                   label='Output',
                   unit='% of heater range',
                   docstring='Specifies heater output in percent of '
                             'the current heater output range.\n'
                             'Note that when the heater is off, '
                             'this parameter will return the value '
                             'of 0.005.',
                   get_parser=float,
                   get_cmd=f'HTR?',
                   set_cmd=False)


class Model_340_Channel(InstrumentChannel):

    SENSOR_STATUSES = {
        0: 'OK',
        1: 'Invalid Reading',
        2: 'Old Reading',
        16: 'Temp Underrange',
        32: 'Temp Overrange',
        64: 'Sensor Units Zero',
        128: 'Sensor Units Overrange'
    }

    def __init__(
        self,
        parent: "Model_340",
        name: str,
        channel: str):

        super().__init__(parent, name)

        self._channel = channel  # Channel on the temperature controller

        self.add_parameter('temperature',
                           get_cmd=f'KRDG? {self._channel}',
                           get_parser=float,
                           label='Temperature',
                           unit='K')

        self.add_parameter('sensor_raw',
                   get_cmd=f'SRDG? {self._channel}',
                   get_parser=float,
                   label='Raw reading',
                   unit='Ohms')

        self.add_parameter('sensor_status',
                   get_cmd=f'RDGST? {self._channel}',
                   get_parser=self._decode_sensor_status,
                   label='Sensor status')

    def _decode_sensor_status(self, sum_of_codes: str) -> str:
        """
        Parses the sum of status code according to the `SENSOR_STATUSES` using
        an algorithm defined in `_get_sum_terms` method.
        Args:
            sum_of_codes
                sum of status codes, it is an integer value in the form of a
                string (e.g. "32"), as returned by the corresponding
                instrument command
        """
        codes = self._get_sum_terms(list(self.SENSOR_STATUSES.keys()),
                                    int(sum_of_codes))
        return ", ".join([self.SENSOR_STATUSES[k] for k in codes])

class Model_340(VisaInstrument):
    """
    Lakeshore Model 340 Temperature Controller Driver
    Note that interaction with the control input (referred to as 'A' in the
    Computer Interface Operation section of the manual) is not implemented.
    """

    channel_name_command: Dict[str, str] = {
        'A': 'A', 'B':'B', 'C':'C', 'D':'D',
    }

    CHANNEL_CLASS = Model_340_Channel

    def __init__(self,
             name: str,
             address: str,
             terminator: str = '\r\n',
             **kwargs: Any
             ) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

        # Allow access to channels either by referring to the channel name
        # or through a channel list, i.e. instr.A.temperature() and
        # instr.channels[0].temperature() refer to the same parameter.
        # Note that `snapshotable` is set to false in order to avoid duplicate
        # snapshotting which otherwise will happen because each channel is also
        # added as a submodule to the instrument.
        channels = ChannelList(self,
                               "TempSensors",
                               self.CHANNEL_CLASS,
                               snapshotable=False)
        for name, command in self.channel_name_command.items():
            channel = self.CHANNEL_CLASS(self, name, command)
            channels.append(channel)
            self.add_submodule(name, channel)
        channels.lock()
        self.add_submodule("channels", channels)

        self.connect_message()

        heaters = {"sample_heater": 1, "analog_1": None, "analog_2": 2}
        for heater_name, loop in heaters.items():
            self.add_submodule(heater_name, Output_340(self, heater_name, loop))
