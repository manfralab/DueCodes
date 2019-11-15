from qcodes import VisaInstrument
from qcodes import validators as vals
import time
import visa


class PS120(VisaInstrument):
    """
    Bare-bones driver for PS120.

    Driver supports only RS232 connections
    """

    _GET_STATUS_CONTROL = {
            0: "Amps, Magnet sweep: fast",
            1: "Tesla, Magnet sweep: fast",
            4: "Amps, Magnet sweep: slow",
            5: "Tesla, Magnet sweep: slow",
            8 : "Amps, (Magnet sweep: unaffected)",
            9 : "Tesla, (Magnet sweep: unaffected)"}

    _GET_STATUS_RAMP = {
            0: "At rest",
            1: "Sweeping",
            2: "Sweep limiting",
            3: "Sweeping & sweep limiting",
            5: "Unknown"}

    _GET_STATUS_SWITCH_HEATER = {
            0: "Off magnet at zero (switch closed)",
            1: "On (switch open)",
            2: "Off magnet at field (switch closed)",
            5: "Heater fault (heater is on but current is low)",
            8: "No switch fitted"}

    _GET_STATUS_REMOTE = {
            0: "Local and locked",
            1: "Remote and locked",
            2: "Local and unlocked",
            3: "Remote and unlocked",
            4: "Auto-run-down",
            5: "Auto-run-down",
            6: "Auto-run-down",
            7: "Auto-run-down"}

    _GET_SUPPLY_STATUS = {
            0: "Normal",
            1: "Quenched",
            2: "Over Heated",
            3: "Warming Up",
            4: "Fault"}

    _GET_LIMIT_STATUS = {
            0: "Normal",
            1: "On positive voltage limit",
            2: "On negative voltage limit",
            3: "Outside negative current limit",
            4: "Outside positive current limit"}

    # _GET_POLARITY_STATUS1 = {
    #         0: "Desired: Positive, Magnet: Positive, Commanded: Positive",
    #         1: "Desired: Positive, Magnet: Positive, Commanded: Negative",
    #         2: "Desired: Positive, Magnet: Negative, Commanded: Positive",
    #         3: "Desired: Positive, Magnet: Negative, Commanded: Negative",
    #         4: "Desired: Negative, Magnet: Positive, Commanded: Positive",
    #         5: "Desired: Negative, Magnet: Positive, Commanded: Negative",
    #         6: "Desired: Negative, Magnet: Negative, Commanded: Positive",
    #         7: "Desired: Negative, Magnet: Negative, Commanded: Negative"}

    # _GET_POLARITY_STATUS2 = {
    #         1: "Negative contactor closed",
    #         2: "Positive contactor closed",
    #         3: "Both contactors open",
    #         4: "Both contactors closed"}

    _SET_ACTIVITY = {
            0: "Hold",
            1: "To set point",
            2: "To zero"}

    _WRITE_WAIT = 100e-3 # seconds

    def __init__(self, name, address, **kwargs):
        """Initializes the Oxford Instruments PS 120 Magnet Power Supply.

        Args:
            name (str)    : name of the instrument
            address (str) : instrument address
            number (int)     : ISOBUS instrument number.
        """
        super().__init__(name, address, terminator='\r', **kwargs)

        self._address = address

        ### Add parameters ###

        # status parameters
        self.add_parameter('remote_status',
                           get_cmd=self._get_remote_status,
                           set_cmd=self._set_remote_status,
                           vals=vals.Ints())
        self.add_parameter('control_mode',
                           get_cmd=self._get_control_mode,
                           set_cmd=self._set_control_mode,
                           vals=vals.Ints())
        self.add_parameter('supply_status',
                           get_cmd=self._get_supply_status)
        self.add_parameter('ramp_status',
                           get_cmd=self._get_ramp_status)
        self.add_parameter('limit_status',
                           get_cmd=self._get_limit_status)
        self.add_parameter('activity',
                           get_cmd=self._get_activity,
                           set_cmd=self._set_activity,
                           vals=vals.Ints())

        # switch parameters
        self.add_parameter('switch_heater',
                           get_cmd=self._get_switch_heater,
                           set_cmd=self._set_switch_heater,
                           vals=vals.OnOff())
        self.add_parameter('heater_current',
                           unit='mA',
                           get_cmd=self._get_heater_current)

        # field parameters
        self.add_parameter('field',
                           unit='T',
                           get_cmd=self._get_field)
        self.add_parameter('field_blocking',
                           unit='T',
                           get_cmd=self._get_field,
                           set_cmd=self.run_to_field_blocking,
                           vals=vals.Numbers(-15, 15))
        self.add_parameter('field_non_blocking',
                           unit='T',
                           get_cmd=self._get_field,
                           set_cmd=self.run_to_field_non_blocking,
                           vals=vals.Numbers(-15, 15))
        self.add_parameter('field_setpoint',
                           unit='T',
                           get_cmd=self._get_field_setpoint,
                           set_cmd=self._set_field_setpoint,
                           vals=vals.Numbers(-15, 15))
        self.add_parameter('sweeprate_field',
                           unit='T/min',
                           get_cmd=self._get_sweeprate_field,
                           set_cmd=self._set_sweeprate_field,
                           vals=vals.Numbers(0, 0.5))

        # unused parameters
        # self.add_parameter('amps_per_tesla',
        #                     initial_value=10.0,
        #                     unit='A/T',
        #                     vals=vals.Numbers(0,25.0))
        # self.add_parameter('polarity',
        #                    get_cmd=self._get_polarity)
        # self.add_parameter('voltage',
        #                    unit='V',
        #                    get_cmd=self._get_voltage)
        # self.add_parameter('voltage_limit',
        #                    unit='V',
        #                    get_cmd=self._get_voltage_limit)
        self.add_parameter('current',
                           unit='A',
                           get_cmd=self._get_current)
        self.add_parameter('magnet_current',
                           unit='A',
                           get_cmd=self._get_magnet_current)
        self.add_parameter('current_setpoint',
                           unit='A',
                           get_cmd=self._get_current_setpoint)
                           # set_cmd=self._set_current_setpoint,
                           # vals=vals.Numbers(-120.0,120.0))
        self.add_parameter('sweeprate_current',
                           unit='A/min',
                           get_cmd=self._get_sweeprate_current)
        #                    set_cmd=self._set_sweeprate_current,
        #                    vals=vals.Numbers(0,3.0))
        self.add_parameter('persistent_current',
                           unit='A',
                           get_cmd=self._get_persistent_current)
        # self.add_parameter('persistent_field',
        #                    unit='T',
        #                    get_cmd=self._get_persistent_field)
        # self.add_parameter('magnet_inductance',
        #                    unit='H',
        #                    get_cmd=self._get_magnet_inductance)
        # self.add_parameter('lead_resistance',
        #                    unit='mOhm',
        #                    get_cmd=self._get_lead_resistance)
        # self.add_parameter('current_limit_lower',
        #                    unit='A',
        #                    get_cmd=self._get_current_limit_lower)
        # self.add_parameter('current_limit_upper',
        #                    unit='A',
        #                    get_cmd=self._get_current_limit_upper)

        # self.add_parameter('trip_field',
        #                    unit='T',
        #                    get_cmd=self._get_trip_field)
        # self.add_parameter('trip_current',
        #                    unit='A',
        #                    get_cmd=self._get_trip_current)

        # setup RS232 communication
        self.visa_handle.set_visa_attribute(
                visa.constants.VI_ATTR_ASRL_STOP_BITS,
                visa.constants.VI_ASRL_STOP_TWO)

    def get_all(self):
        """
        Reads all implemented parameters from the instrument,
        and updates the wrapper.
        """
        self.snapshot(update=True)

    def _float_parser(self, result):
        """
        Take string result and convert to float
        """


    def _execute(self, message):
        """
        Write a command to the device and return the result.

        Args:
            message (str) : write command for the device

        Returns:
            Response from the device as a string.
        """
        self.visa_handle.write(message)
        time.sleep(self._WRITE_WAIT)  # wait for the device to be able to respond
        result = self._read()
        if result.find('?') >= 0:
            raise ValueError("Error: Command %s not recognized" % message)
        else:
            return result.strip()

    def _read(self):
        """
        Reads the total bytes in the buffer and outputs as a string.

        Returns:
            message (str)
        """
        bytes_in_buffer = self.visa_handle.bytes_in_buffer
        with(self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT)):
            mes = self.visa_handle.visalib.read(
                self.visa_handle.session, bytes_in_buffer)
        mes = str(mes[0].decode())
        return mes

    def _identify(self):
        """ Identify the device """
        return self._execute('V')

    def remote(self):
        """Set control to remote and unlocked"""
        self.remote_status(3)

    def local(self):
        """Set control to local and unlocked"""
        self.remote_status(2)

    def close(self):
        """Safely close connection"""
        self.local()
        super().close()

    def get_idn(self):
        """
        Overides the function of Instrument since PS120 does not support `*IDN?`

        This string is supposed to be a comma-separated list of vendor, model,
        serial, and firmware, but semicolon and colon are also common
        separators so we accept them here as well.

        Returns:
            A dict containing vendor, model, serial, and firmware.
        """
        idn = self._identify()
        firmware = idn.split(',')[1].strip()
        idparts = ['Oxford Instruments', 'PS120', None, firmware]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _get_remote_status(self):
        """
        Get remote control status

        Returns:
            result(str) :
            "Local & locked",
            "Remote & locked",
            "Local & unlocked",
            "Remote & unlocked",
            "Auto-run-down",
            "Auto-run-down",
            "Auto-run-down",
            "Auto-run-down"
        """
        result = self._execute('X')
        return self._GET_STATUS_REMOTE[int(result[6])]

    def _set_remote_status(self, mode):
        """
        Set remote control status.

        Args:
            mode(int): Refer to _GET_STATUS_REMOTE for allowed values and
            meanings.
        """
        if mode in self._GET_STATUS_REMOTE.keys():
            self._execute('C%s' % mode)
        else:
            print('Invalid mode inserted: %s' % mode)

    def _get_supply_status(self):
        """
        Get the system status

        Returns:
            result (str) :
            "Normal",
            "Quenched",
            "Over Heated",
            "Warming Up",
            "Fault"
        """
        result = self._execute('X')
        self.log.info('Getting system status')
        return self._GET_SUPPLY_STATUS[int(result[1])]

    def _get_limit_status(self):
        """
        Get the system status

        Returns:
            result (str) :
            "Normal",
            "On positive voltage limit",
            "On negative voltage limit",
            "Outside negative current limit",
            "Outside positive current limit"
        """
        result = self._execute('X')
        self.log.info('Getting system status')
        return self._GET_LIMIT_STATUS[int(result[2])]

    def _get_current(self):
        """
        Demand output current of device

        Returns:
            result (float) : output current in Amp
        """
        result = self._execute('R0')
        return float(result.replace('R', ''))/100.0

    # def _get_voltage(self):
    #     """
    #     Demand measured output voltage of device
    #
    #     Returns:
    #         result (float) : output voltage in Volt
    #     """
    #     self.log.info('Read output voltage')
    #     result = self._execute('R1')
    #     return float(result.replace('R', ''))

    def _get_magnet_current(self):
        """
        Demand measured magnet current of device

        Returns:
            result (float) : measured magnet current in Amp
        """
        self.log.info('Read measured magnet current')
        result = self._execute('R2')
        return float(result.replace('R', ''))/100.0

    def _get_current_setpoint(self):
        """
        Return the set point (target current)

        Returns:
            result (float) : Target current in Amp
        """
        self.log.info('Read set point (target current)')
        result = self._execute('R5')
        return float(result.replace('R', ''))/100.0

    # def _set_current_setpoint(self, current):
    #     """
    #     Set current setpoint (target current)
    #
    #     Args:
    #         current (float) : target current in Amp
    #     """
    #     self.log.info('Setting target current to %s' % current)
    #     self.remote()
    #     self._execute('I%s' % current)
    #     self.local()
    #     self.field_setpoint()

    def _get_sweeprate_current(self):
        """
        Return sweep rate (current)

        Returns:
            result (float) : sweep rate in A/min
        """
        result = self._execute('R6')
        return float(result.replace('R', ''))/100.0

    # def _set_sweeprate_current(self, sweeprate):
    #     """
    #     Set sweep rate (current)
    #
    #     Args:
    #         sweeprate(float) : Sweep rate in A/min.
    #     """
    #     self.remote()
    #     self._execute('S%s' % sweeprate)
    #     self.local()
    #     self.sweeprate_field()

    def _get_field(self):
        """
        Demand output field

        Returns:
            result (float) : magnetic field in Tesla
        """
        self.log.info('Read output field')
        result = self._execute('R7')
        return float(result.replace('R', ''))/1000.0

    def _get_field_setpoint(self):
        """
        Return the set point (target field)

        Returns:
            result (float) : Field set point in Tesla
        """
        result = self._execute('R8')
        return float(result.replace('R', ''))/1000.0

    def _set_field_setpoint(self, field):
        """
        Set the field set point (target field)

        Args:
            field (float) : target field in Tesla
        """
        self.remote()
        self._execute(f'J{100.0*field}')
        self.local()
        self.current_setpoint()

    def _get_sweeprate_field(self):
        """
        Return sweep rate (field)

        Returns:
            result (float) : sweep rate in Tesla/min
        """
        result = self._execute('R9')
        return float(result.replace('R', ''))/1000.0

    def _set_sweeprate_field(self, sweeprate):
        """
        Set sweep rate (field)

        Args:
            sweeprate(float) : Sweep rate in Tesla/min.
        """
        self.remote()
        self._execute(f'T{100.0*sweeprate}')
        self.local()
        self.sweeprate_current()

    # def _get_voltage_limit(self):
    #     """
    #     Return voltage limit
    #
    #     Returns:
    #         result (float) : voltage limit in Volt
    #     """
    #     self.log.info('Read voltage limit')
    #     result = self._execute('R15')
    #     result = float(result.replace('R', ''))
    #     self.voltage.vals = vals.Numbers(-result, result)
    #     return result

    def _get_persistent_current(self):
        """
        Return persistent magnet current

        Returns:
            result (float) : persistent magnet current in Amp
        """
        self.log.info('Read persistent magnet current')
        result = self._execute('R16')
        return float(result.replace('R', ''))/100.0

    # def _get_trip_current(self):
    #     """
    #     Return trip current
    #
    #     Returns:
    #         result (float) : trip current om Amp
    #     """
    #     self.log.info('Read trip current')
    #     result = self._execute('R17')
    #     return float(result.replace('R', ''))
    #
    # def _get_persistent_field(self):
    #     """
    #     Return persistent magnet field
    #
    #     Returns:
    #         result (float) : persistent magnet field in Tesla
    #     """
    #     self.log.info('Read persistent magnet field')
    #     result = self._execute('R18')
    #     return float(result.replace('R', ''))
    #
    # def _get_trip_field(self):
    #     """
    #     Return trip field
    #
    #     Returns:
    #         result (float) : trip field in Tesla
    #     """
    #     self.log.info('Read trip field')
    #     result = self._execute('R19')
    #     return float(result.replace('R', ''))

    def _get_heater_current(self):
        """
        Return switch heater current

        Returns:
            result (float) : switch heater current in milliAmp
        """
        self.log.info('Read switch heater current')
        result = self._execute('R20')
        return float(result.replace('R', ''))

    # def _get_current_limit_upper(self):
    #     """
    #     Return safe current limit, most positive
    #
    #     Returns:
    #         result (float) : safe current limit, most positive in Amp
    #     """
    #     self.log.info('Read safe current limit, most positive')
    #     result = self._execute('R22')
    #     return float(result.replace('R', ''))

    # def _get_current_limit_lower(self):
    #     """
    #     Return safe current limit, most negative
    #
    #     Returns:
    #         result (float) : safe current limit, most negative in Amp
    #     """
    #     self.log.info('Read safe current limit, most negative')
    #     result = self._execute('R21')
    #     return float(result.replace('R', ''))

    # def _get_lead_resistance(self):
    #     """
    #     Return lead resistance
    #
    #     Returns:
    #         result (float) : lead resistance in milliOhm
    #     """
    #     self.log.info('Read lead resistance')
    #     result = self._execute('R23')
    #     return float(result.replace('R', ''))

    # def _get_magnet_inductance(self):
    #     """
    #     Return magnet inductance
    #
    #     Returns:
    #         result (float) : magnet inductance in Henry
    #     """
    #     self.log.info('Read magnet inductance')
    #     result = self._execute('R24')
    #     return float(result.replace('R', ''))

    def _get_activity(self):
        """
        Get the activity of the magnet. Possibilities: Hold, Set point, Zero or Clamp.

        Returns:
            result(str) : "Hold", "Set point", "Zero" or "Clamp".
        """
        result = self._execute('X')
        return self._SET_ACTIVITY[int(result[4])]

    def _set_activity(self, mode):
        """
        Set the activity to Hold, To Set point or To Zero.

        Args:
            mode (int): See _SET_ACTIVITY for values and meanings.
        """
        if mode in self._SET_ACTIVITY.keys():
            self.remote()
            self._execute('A%s' % mode)
            self.local()
        else:
            print('Invalid mode inserted.')

    def hold(self):
        """Set the device activity to Hold"""
        self.activity(0)

    def to_setpoint(self):
        """Set the device activity to "To set point". This initiates a sweep."""
        self.activity(1)

    def to_zero(self):
        """
        Set the device activity to "To zero". This sweeps te magnet back to zero.
        """
        self.activity(2)

    def _get_switch_heater(self):
        """
        Get the switch heater status.

        Returns:
            result(str): See _GET_STATUS_SWITCH_HEATER.
        """
        result = self._execute('X')
        return self._GET_STATUS_SWITCH_HEATER[int(result[8])]

    def _set_switch_heater(self, mode):
        """
        Set the switch heater Off or On. Note: After issuing a command it is necessary to wait
        several seconds for the switch to respond.
        Args:
            mode (int) :
            0 : Off
            1 : On
        """
        message_lookup = {'on':'H1', 'off':'H0'}
        self.remote()
        self._execute(message_lookup[mode])
        print("Setting switch heater... (wait 20s)")
        self.local()
        time.sleep(20)

    # def heater_on(self):
    #     """Switch the heater on, with PSU = Magnet current check"""
    #     current_in_magnet = self.persistent_current()
    #     current_in_leads = self.current()
    #     if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
    #         print('Heater is already on!')
    #     else:
    #         if self.ramp_status() == self._GET_STATUS_RAMP[0]:
    #             if current_in_leads == current_in_magnet:
    #                 self.switch_heater('on')
    #             else:
    #                 print('Current in the leads is not matching persistent current!')
    #         else:
    #             print('Magnet supply not at rest, cannot switch on heater!')
    #     self.switch_heater()

    # def heater_off(self):
    #     """Switch the heater off"""
    #     if (self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[0] or
    #             self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[2]):
    #         print('Heater is already off!')
    #     else:
    #         if self.ramp_status() == self._GET_STATUS_RAMP[0]:
    #             self.switch_heater(0)
    #         else:
    #             print('Magnet is not at rest, cannot switch of the heater!')

    # def set_persistent(self):
    #     """
    #     Puts magnet into persistent mode
    #
    #     Note: After turning of the switch heater we will wait for additional 20
    #     seconds before we put the current to zero. This is done to make sure
    #     that the switch heater is cold enough and becomes superconducting.
    #     """
    #     if self.ramp_status() == self._GET_STATUS_RAMP[0]:
    #         self.heater_off()
    #         print('Waiting for the switch heater to become superconducting')
    #         time.sleep(20)
    #         self.to_zero()
    #         self.get_all()
    #     else:
    #         print('Magnet is not at rest, cannot put it in persistent mode')
    #     self.get_all()

    # def leave_persistent_mode(self):
    #     """
    #     Read out persistent current, match the current in the leads to that current
    #     and switch on heater
    #     """
    #     if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[2]:
    #         field_in_magnet = self.persistent_field()
    #         field_in_leads = self.field()
    #         self.hold()
    #         self.field_setpoint(field_in_magnet)
    #         self.to_setpoint()
    #
    #         while field_in_leads != field_in_magnet:
    #             field_in_leads = self.field()
    #         self.heater_on()
    #         self.hold()
    #
    #     elif self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
    #         print('Heater is already on, so the magnet was not in persistent mode')
    #     elif self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[0]:
    #         print('Heater is off, field is zero. Turning on switch heater.')
    #         self.heater_on()
    #
    #     self.get_all()

    def run_to_field_non_blocking(self, field_value):
        """
        Go to field value

        Args:
            field_value (float): the magnetic field value to go to in Tesla
        """

        if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
            self.hold()
            self.field_setpoint(field_value)
            self.to_setpoint()
        else:
            print('Switch heater is off, cannot change the field.')

    def run_to_field_blocking(self, field_value):
        """
        Go to field value and wait until it's done sweeping.

        Args:
            field_value (float): the magnetic field value to go to in Tesla
        """
        if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
            self.hold()
            self.field_setpoint(field_value)
            self.remote()
            self.to_setpoint()
            magnet_mode = self.ramp_status()
            while magnet_mode != self._GET_STATUS_RAMP[0]:
                magnet_mode = self.ramp_status()
                time.sleep(0.1)
        else:
            print('Switch heater is off, cannot change the field.')
        self.local()



    def _get_control_mode(self):
        """
        Get the mode of the device

        Returns:
            mode(str): See _GET_STATUS_CONTROL.
        """
        result = self._execute('X')
        return self._GET_STATUS_CONTROL[int(result[10])]

    def _get_ramp_status(self):
        """
        Get the sweeping mode of the device

        Returns:
            mode(str): See _GET_STATUS_RAMP.
        """
        result = self._execute('X')
        return self._GET_STATUS_RAMP[int(result[11])]

    def _set_control_mode(self, mode):
        """
        Args:
            mode(int): Refer to _GET_STATUS_CONTROL dictionary for the allowed
            mode values and meanings.
        """
        if mode in self._GET_STATUS_CONTROL.keys():
            self.remote()
            self._execute(f'M{mode}')
            self.local()
        else:
            print('Invalid mode inserted.')

    # def _get_polarity(self):
    #     """
    #     Get the polarity of the output current
    #
    #     Returns:
    #         result (str): See _GET_POLARITY_STATUS1 and _GET_POLARITY_STATUS2.
    #     """
    #     self.log.info('Get device polarity')
    #     result = self._execute('X')
    #     return self._GET_POLARITY_STATUS1[int(result[13])] + \
    #         ", " + self._GET_POLARITY_STATUS2[int(result[14])]
