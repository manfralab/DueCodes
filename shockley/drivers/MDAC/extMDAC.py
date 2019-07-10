"""
DRIVER FOR MDAC V1.5-V1.7
STEVEN WADDY & ANDREW KELLY, QNL, UNIVERSITY OF SYDNEY
Not to be copied, redistributed or reproduced without permission
"""

from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
import math
import re
import collections
import warnings
from fractions import Fraction
import os
import json


class MDACExtChannel(InstrumentChannel):
    """Base Channel Class for MDAC DACs"""

    _WAVEFORM_VALIDATION = vals.Enum('sin', 'tri', 'sqr',
                                     'saw', 'was', 'off')

    def __init__(self, parent, name, channum):
        """
        Args:
            parent (Instrument): The instrument the channel is a part of.
            name (str): the name of the channel
            channum (int): MDAC channel number ([1-64] for populated MDAC)
        """
        super().__init__(parent, name)

        self._cardnum = math.floor((channum-1)/8)+1
        self._channum = channum
        self._min_dac_sample = 0
        self._max_dac_sample = (2**20 - 1)
        self._samplerate = self._parent._samplerate  # Sa/s
        self._LSB = 9.54e-6  # volts
        self._gain = 1
        self._maximum_voltage = 5
        self._minimum_voltage = -5
        self._divider_ratio = 0.01
        self._arb_data_max_samples = 32
        self._default_ramprate = 0.001  # V/s
        self._autosync = True  # sync after waveform update

        dac_config = self._get_dac_config()
        self._wave_max_samples = int(dac_config[1])
        self._ramp_max_samples = int(dac_config[3])
        self._wave_max_seconds = float(dac_config[2])
        self._ramp_max_seconds = float(dac_config[4])

        # channel paramaters
        self.add_parameter(
            'voltage',
            label='Channel {} voltage'.format(self._channum),
            unit='V',
            set_cmd=self._set_voltage,
            get_cmd=self._get_voltage,
            vals=vals.Numbers(-5, 5),
            docstring='current output voltage [volts], set immediately'
            )

        self.add_parameter(
            'voltage_raw',
			label='Channel {} voltage_raw'.format(self._channum),
            unit='LSB',
            set_cmd=self._set_dac_raw,
            get_cmd=self._get_dac_raw,
            vals=vals.Ints(self._min_dac_sample, self._max_dac_sample),
            docstring='current output voltage [raw values], set immediately'
            )

        self.add_parameter(
            'amplitude',
            label='Channel {} amplitude'.format(self._channum),
            unit='Vpp',
            get_cmd=self._get_amplitude,
            set_cmd=self._set_otf_amplitude,
            vals=vals.Numbers(0, 10),
            docstring='current waveform\'s peak-to-peak amplitude (zero for DC)'
            )

        self.add_parameter(
            'offset',
            label='Channel {} offset'.format(self._channum),
            unit='V',
            get_cmd=self._get_offset,
            set_cmd=self._set_otf_offset,
            vals=vals.Numbers(-5, 5),
            docstring='current waveform\'s offset'
            )

        self.add_parameter(
            'frequency',
            label='Channel {} frequency'.format(self._channum),
            unit='Hz',
            get_cmd=self._get_frequency,
            vals=vals.Numbers(0, round(self._samplerate/2)),
            docstring='current waveform\'s frequency'
            )

        self.add_parameter(
            'period',
            label='Channel {} period'.format(self._channum),
            unit='s',
            get_cmd=self._get_period,
            vals=vals.Numbers(1/(self._samplerate/2), 10),
            docstring='current waveform\'s period'
            )

        self.add_parameter(
            'phase',
            label='Channel {} phase offset'.format(self._channum),
            unit='Deg',
            get_cmd=self._get_phase_offset,
            set_cmd=self._set_phase_offset,
            vals=vals.Numbers(0, 360),
            docstring='current waveform\'s phase offset'
            )

        self.add_parameter(
            'waveform',
            label='Channel {} waveform'.format(self._channum),
            get_cmd=self._get_waveform,
            get_parser=self._lowercase_parser,
            vals=self._WAVEFORM_VALIDATION,
            docstring='current waveform\'s type'
            )

        self.add_parameter(
            'default_ramprate',
            label='Channel {} default ramp-rate'.format(self._channum),
            unit='V/s',
            get_cmd=self._get_default_ramprate,
            set_cmd=self._set_default_ramprate,
            vals=vals.Numbers(0, 100000),
            docstring='default ramp-rate used for the ramp command'
        )

        self.add_parameter(
            'ramp_rate',
            label='Channel {} ramp-rate'.format(self._channum),
            unit='V/s',
            get_cmd=self._get_ramprate,
            docstring='returns the current ramp rate\n'
                      'or None if there is no active ramp'
            )

        self.add_parameter(
            'ramp_destination',
            label='Channel {} ramp end-point'.format(self._channum),
            unit='V',
            get_cmd=self._get_ramp_destination,
            docstring='returns the current ramp destination\n'
                      'or None if there is no active ramp'
            )

        self.add_parameter(
            'autosync',
            label='Channel {} waveform auto resync'.format(self._channum),
            get_cmd=self._get_autosync,
            set_cmd=self._set_autosync,
            vals=vals.Bool(),
            docstring='True, issues a resync command after a waveform update\n'
                      'False, a resync must be issued using MDAC.sync()'
            )

        if self._parent._hardware_limits:
            self.add_parameter(
                'limit_max',
                label='Channel {} maximum voltage limit'.format(self._channum),
                unit='V',
                get_cmd=self._get_maxlimit,
                set_cmd=self._set_maxlimit,
                vals=vals.Numbers(-5, 5),
                docstring='maximum voltage hardware limit'
                )

            self.add_parameter(
                'limit_min',
                label='Channel {} maximum voltage limit'.format(self._channum),
                unit='V',
                get_cmd=self._get_minlimit,
                set_cmd=self._set_minlimit,
                vals=vals.Numbers(-5, 5),
                docstring='minimum voltage hardware limit'
                )

            self.add_parameter(
                'limit_rate',
                label='Channel {} maximum rate limit'.format(self._channum),
                unit='V/s',
                get_cmd=self._get_ratelimit,
                set_cmd=self._set_ratelimit,
                vals=vals.Numbers(0.2, 203504),
                docstring='maximum rate of change, hardware limit'
                )

        self.add_function(
            'attach_trigger',
            call_cmd=self._attach_trigger,
            docstring='Assigns channel as trigger source'
            )

        self.add_function(
            'smc_to_dac',
            call_cmd=self._smc_to_dac)

        self.add_function(
            'microd_to_dac',
            call_cmd=self._microd_to_dac)

        self.add_function(
            'microd_float',
            call_cmd=self._microd_float)

        self.add_function(
            'microd_to_bus',
            call_cmd=self._microd_to_bus)

        self.add_function(
            'microd_to_smc',
            call_cmd=self._microd_to_smc)

        self.add_function(
            'smc_to_bus',
            call_cmd=self._smc_to_bus)

        self.add_function(
            'terminate',
            call_cmd=self._terminate)

        self.add_function(
            'smc_float',
            call_cmd=self._smc_float)

        self.add_function(
            'smc_ground',
            call_cmd=self._smc_ground)

        self.add_function(
            'microd_ground',
            call_cmd=self._microd_ground)


        self._update_voltage_range()


    def _smc_to_dac(self):
        self.bus('open')
        self.gnd('open')
        self.microd('open')
        self.dac_output('close')
        self.smc('close')

    def _microd_to_dac(self):
        self.smc('open')
        self.bus('open')
        self.gnd('open')
        self.dac_output('close')
        self.microd('close')

    def _microd_float(self):
        self.dac_output('open')
        self.smc('open')
        self.bus('open')
        self.gnd('open')
        self.microd('close')

    def _microd_to_bus(self):
        self.dac_output('open')
        self.gnd('open')
        self.smc('open')
        self.bus('close')
        self.microd('close')

    def _microd_to_smc(self):
        self.dac_output('open')
        self.gnd('open')
        self.bus('open')
        self.smc('close')
        self.microd('close')

    def _smc_to_bus(self):
        self.dac_output('open')
        self.gnd('open')
        self.microd('open')
        self.bus('close')
        self.smc('close')

    def _terminate(self):
        self.smc('open')
        self.microd('open')
        self.bus('open')
        self.gnd('open')
        self.dac_output('open')

    def _smc_float(self):
        self.dac_output('open')
        self.bus('open')
        self.gnd('open')
        self.microd('open')
        self.smc('close')

    def _smc_ground(self):
        self.dac_output('open')
        self.bus('open')
        self.microd('open')
        self.gnd('close')
        self.smc('close')

    def _microd_ground(self):
        self.dac_output('open')
        self.smc('open')
        self.bus('open')
        self.gnd('close')
        self.microd('close')


    def channel_number(self):
        """ DAC channel number corresponding to SMC channel numbers"""
        return self._channum

    def _attach_trigger(self):
        cmd='DAC:{}:TRIGGER SET'.format(self._channum)
        self.write(cmd)

    def _get_dac(self):
        resp = self.ask('DAC:{}:OUTPUT ?'.format(self._channum))
        volt = float(resp.split(',')[0])
        raw = int(resp.split(',')[1], 16)
        return [volt, raw]

    def _set_voltage(self, voltage):
        cmd = 'DAC:{}:OUTPUT {}'.format(self._channum, voltage)
        self.write(cmd)

    def _get_voltage(self):
        return self._get_dac()[0]

    def _set_dac_raw(self, dac):
        cmd = 'DAC:{}:OUTPUT RAW{}'.format(self._channum, dac)
        self.write(cmd)

    def _get_dac_raw(self):
        return self._get_dac()[1]

    def _set_autosync(self, autosync):
        self._autosync = autosync

    def _get_autosync(self):
        return self._autosync

    def _set_default_ramprate(self, rate):
        self._default_ramprate = rate

    def _get_default_ramprate(self):
        return self._default_ramprate

    def _get_ramprate(self):
        preset = self._get_preset()
        if preset[0] != 'ramp':
            rate = None
        else:
            period = float(preset[1])
            start_volts = float(preset[3])
            dest_volts = float(preset[5])
            rate = abs((dest_volts - start_volts) / period)
        return rate

    def _get_ramp_destination(self):
        preset = self._get_preset()
        if preset[0] != 'ramp':
            dest = None
        else:
            dest = float(preset[5])
        return dest

    def _ramp(self, voltage, rate=None):
        if rate is None:
            rate = self._default_ramprate
        SR = self._samplerate
        LSB = self._LSB
        rawrate = round(abs(rate)/(LSB*SR), 3)
        if rawrate < 0.001:
            rawrate = 0.001
        self.write('DAC:{}:PRESET RAMP,RATERAW{},@,{},S1'.format(
            self._channum, rawrate, voltage))

    def _card_voltages(self):
        # maximum range clipped in HW REV 1
        # cliped to ~ (13V, -10V)
        r = self.ask('SLAVE:{}:VOLTAGES ?'.format(self._cardnum))
        v3 = round(float(r.split(',')[5]), 2)
        v15 = round(float(r.split(',')[3]), 2)
        v15m = round(float(r.split(',')[1]), 2)
        return tuple([v3, v15, v15m])

    def _card_voltages_raw(self):
        r = self.ask('SLAVE:{}:VOLTAGES ?'.format(self._cardnum))
        i3 = int(r.split(',')[4], 16)
        i15 = int(r.split(',')[2], 16)
        i15m = int(r.split(',')[0], 16)
        return tuple([i3, i15, i15m])

    def ramp(self, destn_voltage, ramp_rate=None):
        """
        Helper function to set a ramp from the current voltage to a
        specified destination.  If a ramp-rate is not specified, the
        default is used.

        Args:
            destn_voltage = ramp destination voltage (Volts)
            ramp_rate = ramp rate (Volts/Sec) - (default = default_ramp_rate)
        """
        if ramp_rate == None:
            ramp_rate = self._default_ramprate
        return self._ramp(destn_voltage, ramp_rate)

    def block(self):
        """
        Blocks until channel is not ramping
        """
        self.parent.wait_ramps(self)

    def awg_sine(self, frequency, amplitude, offset=0, phase=0):
        """
        Helper function to set the waveform
        to a sine wave at a given frequency, amplitude and offset

        Args:
            frequency - [Hz]
            amplitude - [Vpp]
            offset - [V]
            phase - [Deg]
        """
        period = 1/frequency
        delay = period * phase/360
        self.write('DAC:{}:PRESET SIN,{},{},{},{}'.format(
            self._channum, period, offset, amplitude, delay))
        if self._autosync:
            self.parent.sync()

    def awg_sawtooth(self, frequency, amplitude, offset=0, phase=0):
        """
        Helper function to set the waveform
        to a rising sawtooth wave at a given frequency, amplitude and offset

        Args:
            frequency - [Hz]
            amplitude - [Vpp]
            offset - [V]
            phase - [Deg]
        """
        period = 1/frequency
        delay = period * phase/360
        self.write('DAC:{}:PRESET SAW,{},{},{},{}'.format(
            self._channum, period, offset, amplitude, delay))
        if self._autosync:
            self.parent.sync()

    def awg_sawtooth_falling(self, frequency, amplitude, offset=0, phase=0):
        """
        Helper function to set the waveform
        to a falling sawtooth wave at a given frequency, amplitude and offset

        Args:
            frequency - [Hz]
            amplitude - [Vpp]
            offset - [V]
            phase - [Deg]
        """
        period = 1/frequency
        delay = period * phase/360
        self.write('DAC:{}:PRESET WAS,{},{},{},{}'.format(
            self._channum, period, offset, amplitude, delay))
        if self._autosync:
            self.parent.sync()

    def awg_square(self, frequency, amplitude, offset=0, phase=0):
        """
        Helper function to set the waveform
        to a square wave at a given frequency, amplitude and offset

        Args:
            frequency - [Hz]
            amplitude - [Vpp]
            offset - [V]
            phase - [Deg]
        """
        period = 1/frequency
        delay = period * phase/360
        self.write('DAC:{}:PRESET SQR,{},{},{},{}'.format(
            self._channum, period, offset, amplitude, delay))
        if self._autosync:
            self.parent.sync()

    def awg_triangle(self, frequency, amplitude, offset=0, phase=0):
        """
        Helper function to set the waveform
        to a triangle wave at a given frequency, amplitude and offset

        Args:
            frequency - [Hz]
            amplitude - [Vpp]
            offset - [V]
            phase - [Deg]
        """
        period = 1/frequency
        delay = period * phase/360
        self.write('DAC:{}:PRESET TRI,{},{},{},{}'.format(
            self._channum, period, offset, amplitude, delay))
        if self._autosync:
            self.parent.sync()

    def awg_off(self):
        """Helper function to turn the waveform off"""
        self.write('DAC:{}:PRESET NONE'.format(self._channum))
        if self._autosync:
            self.parent.sync()

    def _volts_to_raw(self, volts):
        if volts > 0:
            dac_range = self._max_dac_sample - self._zero_volts_dac
            volts_per_dac = self._maximum_voltage / dac_range
        else:
            dac_range = self._min_dac_sample - self._zero_volts_dac
            volts_per_dac = self._minimum_voltage / dac_range
        dac = (volts / volts_per_dac) + self._zero_volts_dac
        return int(dac + 0.5)


    def awg_arbitrary_wave(self, arb_data, phase=0):
        """
        Helper function to set an arbitrary waveform

        Args:
            arb_data - an array of 0V offsets in Volts
            phase - [Deg]
        """
        sample_count = len(arb_data)
        if sample_count > self._wave_max_samples:
            warn_str = 'Maximum wave length ({} samples) exceeded'\
                       .format(self._wave_max_samples)
            self.parent._handle_error(warn_str)
        else:
            arb_data_raw = []
            for volts in arb_data:
                sample = self._volts_to_raw(volts)
                offset = sample - self._zero_volts_dac
                arb_data_raw.append(offset)
            self.awg_arbitrary_wave_raw(arb_data_raw, phase)

    def awg_arbitrary_wave_raw(self, arb_data, phase=0):
        """
        Helper function to set an arbitrary waveform

        Args:
            arb_data - an array of raw DAC data
            phase - [Deg]
        """
        sample_count = len(arb_data)
        self.write('DAC:{}:PRESET OFF,RAW{}'
                   ''.format(self._channum, sample_count))
        arb_data_offset = 0
        arb_data_str = ''
        arb_data_num_samples = 0
        for offset in arb_data:
            sample = self._zero_volts_dac + offset
            # Ensure sample values don't wrap.
            sample = min(max(sample, self._min_dac_sample),
                         self._max_dac_sample)
            sample_str = '{:05X}'.format(sample)
            arb_data_str += sample_str
            arb_data_num_samples += 1
            if arb_data_num_samples == self._arb_data_max_samples:
                # This is the max data that can be sent in a single command
                self.write('DAC:{}:PRESET ARB_DATA,{},{}'
                           .format(self._channum, arb_data_offset,
                                   arb_data_str))
                arb_data_offset += arb_data_num_samples
                arb_data_str = ''
                arb_data_num_samples = 0
        if len(arb_data_str) > 0:
            self.write('DAC:{}:PRESET ARB_DATA,{},{}'
                       .format(self._channum, arb_data_offset, arb_data_str))
        self.write('DAC:{}:PRESET ARB'.format(self._channum))
        self._set_phase_offset(phase)
        if self._autosync:
            self.parent.sync()

    def _get_awg_arbitrary_wave_raw(self):
        """
        Internal helper function to get the data from an arbitrary waveform
        Returns:
            A tuple of length equal to the number of samples in the waveform
        """
        arb_data = []
        samples_remaining = self._get_period_raw()
        num_samples_retrieved = 0
        while (samples_remaining > 0):
            # Never ask for > 512 samples
            request_num_samples = min(samples_remaining, 512)
            arb_data_str = self.parent.query(
                'DAC:{}:PRESET ARB_DATA,{}?{}'
                    .format(self._channum, num_samples_retrieved,
                            request_num_samples))
            for sample_index in range(0, request_num_samples):
                arb_data_offset = sample_index * 5
                sample_str = arb_data_str[arb_data_offset : arb_data_offset+5]
                sample = int(sample_str, 16)
                offset = sample - self._zero_volts_dac
                arb_data.append(offset)
            num_samples_retrieved += request_num_samples
            samples_remaining -= request_num_samples
        return arb_data

    def _get_preset(self):
        """
        Internal helper function to get the active preset
        Returns:
            _get_preset()[0] is the waveform
            _get_preset()[1,2] is the period in seconds, ticks
            _get_preset()[3,4] is the offset/ramp_start in volts, dac units
            _get_preset()[5,6] is the amplitude/ramp_destn in volts, dac units
            _get_preset()[7] is the number of shots
            _get_preset()[8,9] is the phase offset in seconds, ticks
        """
        response = self.ask('DAC:{}:PRESET ?'.format(self._channum))
        # [PRESET, period_secs, period_samples, offs_V, offset_DAC, amp_V,
        #  amp_DAC, num_shots, phase_secs, phase_samples]
        if response.startswith('NONE'):
            # d = self._get_dac()
            resp = self.ask('DAC:{}:OUTPUT ?'.format(self._channum))
            volt_str = resp.split(',')[0]
            raw_str = resp.split(',')[1]
            ret =  ['off', '0', '0', volt_str, raw_str, '0', '0', '0', '0', '0']
        elif response.startswith('DC'):
            # d = self._get_dac()
            resp = self.ask('DAC:{}:OUTPUT ?'.format(self._channum))
            volt_str = resp.split(',')[0]
            raw_str = resp.split(',')[1]
            ret =  ['dc', '0', '0', volt_str, raw_str, '0', '0', '0', '0', '0']
        else:
            ret = response.lower().split(',')

        return ret

    def _set_preset(self, preset):
        """
        Internal helper function to set a preset
        Args:
            preset is a tuple comsisting of:
            - preset[0] is the waveform
            - preset[1,2] is the period in seconds, ticks
            - preset[3,4] is the offset/ramp_start in volts, dac units
            - preset[5,6] is the amplitude/ramp_destination in volts, dac units
            - preset[7] is the number of shots
            - preset[8,9] is the phase offset in seconds, ticks
        """
        if preset[0] == 'off':
            preset_cmd = 'OFF'
        else:
            preset_cmd = preset[0].upper()
            # period
            if preset[1] != None:
                preset_cmd += ',{}'.format(preset[1])
            else:
                preset_cmd += ',RAW{}'.format(preset[2])
            # offset/ramp_start
            if preset[3] != None:
                preset_cmd += ',{}'.format(preset[3])
            else:
                preset_cmd += ',RAW{}'.format(preset[4])
            # amplitude/ramp_destination
            if preset[5] != None:
                preset_cmd += ',{}'.format(preset[5])
            else:
                preset_cmd += ',RAW{}'.format(preset[6])
            # number of shots
            if preset[7] != None:
                preset_cmd += ',S{}'.format(preset[7])
            # phase
            if preset[8] != None:
                preset_cmd += ',{}'.format(preset[8])
            else:
                preset_cmd += ',RAW{}'.format(preset[9])

        self.write('DAC:{}:PRESET {}'.format(self._channum, preset_cmd))
        if self._autosync:
            self.parent.sync()

    def _get_otf_scaler(self):
        response = self.ask('DAC:{}:PRESET:SCALER ?'.format(self._channum))
        return float(response.split(',')[0])

    def _set_otf_scaler(self, amplitude):
        preset = self._get_preset()
        waveform = self.waveform()
        if (waveform == 'off') or (waveform == 'dc'):
            self.parent._handle_error('Illegal operation - waveform: {}'
                                      .format(waveform))
        else:
            self.write('DAC:{}:PRESET:SCALER {}'
                       .format(self._channum, amplitude))

    def _get_otf_amplitude(self):
        response = self.ask('DAC:{}:PRESET:AMPLITUDE ?'.format(self._channum))
        return float(response.split(',')[0])

    def _get_otf_amplitude_raw(self):
        response = self.ask('DAC:{}:PRESET:AMPLITUDE ?'.format(self._channum))
        return int(response.split(',')[1], 16)

    def _set_otf_amplitude(self, amplitude):
        preset = self._get_preset()
        waveform = self.waveform()
        if (waveform == 'off') or (waveform == 'dc') or (waveform == 'arb'):
            self.parent._handle_error('Illegal operation - waveform: {}'
                                      .format(waveform))
        else:
            self.write('DAC:{}:PRESET:AMPLITUDE {}'
                       .format(self._channum, amplitude))

    def _set_otf_amplitude_raw(self, amplitude):
        preset = self._get_preset()
        waveform = self.waveform()
        if (waveform == 'off') or (waveform == 'dc') or (waveform == 'arb'):
            self.parent._handle_error('Illegal operation - waveform: {}'
                                      .format(waveform))
        else:
            self.write('DAC:{}:PRESET:AMPLITUDE RAW{}'
                       .format(self._channum, amplitude))

    def _get_otf_offset(self):
        response = self.ask('DAC:{}:PRESET:OFFSET ?'.format(self._channum))
        return float(response.split(',')[0])

    def _get_otf_offset_raw(self):
        response = self.ask('DAC:{}:PRESET:OFFSET ?'.format(self._channum))
        return int(response.split(',')[1], 16)

    def _set_otf_offset(self, offset):
        self.write('DAC:{}:PRESET:OFFSET {}'.format(self._channum, offset))

    def _set_otf_offset_raw(self, offset):
        self.write('DAC:{}:PRESET:OFFSET RAW{}'.format(self._channum, offset))

    def _get_dac_config(self):
        """
        Internal helper function to get the current dac configuration
        Returns:
            _get_dac_config[0] is DAC update rate in Hz
            _get_dac_config[1,2] is max length (samples,secs) for all waveforms
            _get_dac_config[3,4] is max length (samples,secs) for RAMPs
            The remaining fields are _reserved_
        """
        if self._cardnum == 0:
            cmd = 'MASTER:DAC_SPECS ?'
        else:
            cmd = 'SLAVE:{}:DAC_SPECS ?'.format(self._cardnum)
        response = self.parent.query(cmd)
        dac_config = response.split(',')
        return dac_config

    def _get_calibration(self):
        """
		Internal helper function that returns a dictionary representing the
        calibration data for the channel
		"""
        response = self.ask('DAC:{}:CALIBRATION ?'.format(self._channum))
        split = response.splitlines()
        divider_filter_cal = split[4].split(',')
        cal_data_dict = {}
        cal_data_dict['min_volt'] = float(split[1].split(',')[1])
        cal_data_dict['max_volt'] = float(split[3].split(',')[1])
        cal_data_dict['zero_code'] = int(split[2].split(',')[0], 16)
        cal_data_dict['filter_1_cutoff'] = float(divider_filter_cal[2])
        cal_data_dict['filter_2_cutoff'] = float(divider_filter_cal[3])

        cal_data_dict['divider_attenuation'] = float(divider_filter_cal[1])
        cal_data_dict['output_gain'] = float(divider_filter_cal[0])
        return cal_data_dict

    def _set_calibration(self, cal_dict):
        """
        Internal helper function that accepts a  dictionary in the same form
		as the one given by calling _get_calibration(), to alter the calibration
		data of the DAC
        """
        cal_string = ''
        cal_string += '0,{},'.format(cal_dict['min_volt'])
        cal_string += '{},0,'.format(cal_dict['zero_code'])
        cal_string += '1048575,{}'.format(cal_dict['max_volt'])

        self.write('DAC:{}:CALIBRATION {}'.format(self._channum, cal_string))
        self.write('DAC:{}:CALIBRATION:GAIN {}'.format(
            self._channum, cal_dict['output_gain']))

        divider_fraction = Fraction(
            cal_dict['divider_attenuation']).limit_denominator(65535)
        self.write('DAC:{}:CALIBRATION:DIVIDER {},{}'.format(
            self._channum,
            divider_fraction.numerator,
            divider_fraction.denominator))

        self._update_voltage_range()
        self._update_voltage_gain()

    def _store_calibration(self):
        """
		Internal helper function that stores the calibration to NV memory
        NOTE: stores the entire card's channel calibration data to memory
		"""
        self.write('DAC:{}:CALIBRATION STORE'.format(self._channum))

    def _get_limits(self):
        s = self.ask('DAC:{}:LIMITS ?'.format(self._channum))
        s = s.split(',')
        ret = [None] * 6
        ret[0] = float(s[0])
        ret[1] = float(s[2])
        ret[2] = float(s[4]) * self._samplerate
        ret[3] = float(s[6])
        ret[4] = float(s[8])
        ret[5] = float(s[10]) * self._samplerate
        return ret

    def _set_limits(self, minlimit='', maxlimit='', ratelimit=''):
        maxdelta = ''
        if ratelimit != '':
            maxdelta = ratelimit / (self._LSB * self._samplerate)
            maxdelta = max(maxdelta, 1)
            maxdelta = 'RAW{}'.format(int(maxdelta + 0.5))
        self.write('DAC:{}:LIMITS {},{},{}'.format(
            self._channum, minlimit, maxlimit, maxdelta))

    def _store_limits(self):
        """
        Internal helper function that stores the HW limits to NV memory
        NOTE: stores the entire card's channel limits data to memory
        """
        self.write('DAC:{}:LIMITS:STORE'.format(self._channum))

    def _reset_limits(self):
        self.write('DAC:{}:LIMITS:RESET'.format(self._channum))

    def _set_maxlimit(self, limit):
        self._set_limits(maxlimit=limit)

    def _set_minlimit(self, limit):
        self._set_limits(minlimit=limit)

    def _set_ratelimit(self, limit):
        self._set_limits(ratelimit=limit)

    def _get_maxlimit(self):
        return self._get_limits()[1]

    def _get_minlimit(self):
        return self._get_limits()[0]

    def _get_ratelimit(self):
        return self._get_limits()[2]

    def _get_waveform(self):
        preset = self._get_preset()
        return preset[0]

    def _get_amplitude(self):
        preset = self._get_preset()
        if (preset[0] == 'off'):
            ret = 0
        elif (preset[0] == 'ramp'):
            ret = 0
        elif (preset[0] == 'arb'):
            ret = self._get_otf_amplitude()
        else:
            ret = float(preset[5])
        return ret

    def _get_amplitude_raw(self):
        preset = self._get_preset()
        if (preset[0] == 'off'):
            ret = 0
        elif (preset[0] == 'ramp'):
            ret = 0
        elif (preset[0] == 'arb'):
            ret = self._get_otf_amplitude_raw()
        else:
            ret = int(preset[6], 16)
        return ret

    def _get_offset(self):
        preset = self._get_preset()
        if (preset[0] == 'off'):
            ret = self._get_voltage()
        elif (preset[0] == 'ramp'):
            ret = 0
        elif  (preset[0] == 'arb'):
            ret = self._get_otf_offset()
        else:
            ret = float(preset[3])
        return ret

    def _get_offset_raw(self):
        preset = self._get_preset()
        if (preset[0] == 'off'):
            ret = self._get_dac_raw()
        elif (preset[0] == 'ramp'):
            ret = 0
        elif (preset[0] == 'arb'):
            ret = self._get_otf_offset_raw()
        else:
            ret = int(preset[4], 16)
        return ret

    def _get_period(self):
        preset = self._get_preset()
        if preset[0] == 'off':
            ret = 0
        else:
            ret = float(preset[1])
        return ret

    def _get_period_raw(self):
        preset = self._get_preset()
        if preset[0] == 'off':
            ret = 0
        else:
            ret = int(preset[2])
        return ret

    def _get_frequency(self):
        period = self._get_period()
        if period == 0:
            ret = 0.0
        else:
            ret = 1.0/period
        return ret

    def _set_phase_offset(self, phase):
        period = self._get_period()
        delay = period * (phase / 360)
        self.write('DAC:{}:PRESET:PHASE {}'.format(self._channum, delay))

    def _get_phase_delay(self):
        preset = self._get_preset()
        if preset[0] == 'off':
            time = 0
        elif preset[0] == 'arb':
            time = float(preset[4])
        else:
            time = float(preset[8])
        return time

    def _get_phase_delay_raw(self):
        preset = self._get_preset()
        if preset[0] == 'off':
            time = 0
        elif preset[0] == 'arb':
            time = int(preset[5])
        else:
            time = int(preset[9])
        return time

    def _get_phase_offset(self):
        period = self._get_period()
        if period == 0:
            ret = 0
        else:
            time = self._get_phase_delay()
            phase = 360 * (time / period)
            ret = round(phase, 2)
        return ret

    def _lowercase_parser(self, s):
        s = s.lower()
        return s

    def _uppercase_parser(self, s):
        s = s.upper()
        return s

    def _update_voltage_range(self):
        cal = self._get_calibration()
        self._minimum_voltage = cal['min_volt']
        self._zero_volts_dac = cal['zero_code']
        self._maximum_voltage = cal['max_volt']
        self._divider_ratio = cal['divider_attenuation']

    def _update_voltage_gain(self):
        maximum_voltage = self._maximum_voltage
        minimum_voltage = self._minimum_voltage

        if self.divider() == 'on':
            self._gain = self._divider_ratio
            self._LSB = 0.1 / (2**20)
        else:
            self._gain = 1
            self._LSB = 10 / (2**20)

        # calculate new min / max voltage
        maximum_voltage *= self._gain
        minimum_voltage *= self._gain
        # calculate new LSB
        v = vals.Numbers(minimum_voltage, maximum_voltage)
        self.voltage.vals = v
        self.offset.vals = v
        self.ramp.vals = v

        v = vals.Numbers(0, maximum_voltage - minimum_voltage)
        self.amplitude.vals = v


class MDACChannel(MDACExtChannel):
    """Channel Class for MDAC"""

    def __init__(self, parent, name, channum):
        """
        Args:
            parent (Instrument): The instrument the channel is a part of.
            name (str): the name of the channel
            channum (int): MDAC channel number ([1-64] for populated MDAC)
        """
        super().__init__(parent, name, channum)

        self.add_parameter(
            'divider',
            label='Channel {} divider'.format(self._channum),
            set_cmd=self._set_divider,
            get_cmd='DAC:{}:RELAYS:DIVIDER ?'.format(self._channum),
            vals=vals.OnOff(),
            val_mapping={'on': 1, 'off': 0},
            docstring='dac output voltage divider\n'
                      '(currently voltage ranges not updated to new range)'
        )

        self.add_parameter(
            'filter',
            label='Channel {} filter'.format(self._channum),
            set_cmd='DAC:{}:FILTER {{}}'.format(self._channum),
            get_cmd='DAC:{}:FILTER ?'.format(self._channum),
            vals=vals.Enum(1, 2),
            get_parser=int,
            docstring='dac output filter\n'
                      '1 -> (~ 1000 KHz LPF)\n'
                      '2 -> (~ 10 Hz LPF)'
        )

        self.add_parameter(
            'microd',
            label='Channel {} Micro-D relay'.format(self._channum),
            set_cmd='DAC:{}:RELAYS:MAIN {{}}'.format(self._channum),
            get_cmd='DAC:{}:RELAYS:MAIN ?'.format(self._channum),
            val_mapping={'open': 0, 'close': 1},
            docstring='micro-D output relay'
        )

        self.add_parameter(
            'dac_output',
            label='Channel {} output relay'.format(self._channum),
            set_cmd='DAC:{}:RELAYS:DAC_OUTPUT {{}}'.format(self._channum),
            get_cmd='DAC:{}:RELAYS:DAC_OUTPUT ?'.format(self._channum),
            val_mapping={'open': 0, 'close': 1},
            docstring='dac output relay onto signal routing common point'
        )

        self.add_parameter(
            'smc',
            label='Channel {} SMC relay'.format(self._channum),
            set_cmd='DAC:{}:RELAYS:SMC {{}}'.format(self._channum),
            get_cmd='DAC:{}:RELAYS:SMC ?'.format(self._channum),
            val_mapping={'open': 0, 'close': 1},
            docstring='SMC output relay'
        )

        self.add_parameter(
            'bus',
            label='Channel {} BUS relay'.format(self._channum),
            set_cmd='DAC:{}:RELAYS:BUS {{}}'.format(self._channum),
            get_cmd='DAC:{}:RELAYS:BUS ?'.format(self._channum),
            val_mapping={'open': 0, 'close': 1},
            docstring='signal bus relay'
        )

        self.add_parameter(
            'gnd',
            label='Channel {} GND relay'.format(self._channum),
            set_cmd='DAC:{}:RELAYS:TERMINATE {{}}'.format(self._channum),
            get_cmd='DAC:{}:RELAYS:TERMINATE ?'.format(self._channum),
            val_mapping={'open': 0, 'close': 1},
            docstring='internal grounding relay'
        )

        self._update_voltage_gain()

    def _set_divider(self, cmd):
        self.write('DAC:{}:RELAYS:DIVIDER {}'.format(self._channum, cmd))
        self._update_voltage_gain()

    def _update_voltage_range(self):
        cal = self._get_calibration()
        self._minimum_voltage = cal['min_volt']
        self._zero_volts_dac = cal['zero_code']
        self._maximum_voltage = cal['max_volt']
        self._divider_ratio = cal['divider_attenuation']

    def _update_voltage_gain(self):
        maximum_voltage = self._maximum_voltage
        minimum_voltage = self._minimum_voltage

        if self.divider() == 'on':
            self._gain = self._divider_ratio
            self._LSB = 0.1 / (2**20)
        else:
            self._gain = 1
            self._LSB = 10 / (2**20)

        # calculate new min / max voltage
        maximum_voltage *= self._gain
        minimum_voltage *= self._gain
        # calculate new LSB
        v = vals.Numbers(minimum_voltage, maximum_voltage)
        self.voltage.vals = v
        self.offset.vals = v

        v = vals.Numbers(0, maximum_voltage - minimum_voltage)
        self.amplitude.vals = v


class Trigger(InstrumentChannel):
    """Dedicated trigger channel class for MDAC"""

    _TRIGGER_DIRECTION_VALIDATION = vals.Enum('up', 'down')

    def __init__(self, parent, name, trignum):
        """
        Args:
            parent (Instrument): The instrument the channel is a part of.
            name (str): the name of the trigger
            trignum (int): MDAC trigger number ([0] for v1.5 MDAC firmware)
        """
        super().__init__(parent, name)

        self._trignum = trignum
        self._autosync = True

        self.add_parameter(
            'autosync',
            label='Trigger {} waveform auto resync'.format(self._trignum),
            get_cmd=self._get_autosync,
            set_cmd=self._set_autosync,
            vals=vals.Bool(),
            docstring='True, issues a resync command after a trigger update\n'
                      'False, a resync must be issued using MDAC.sync()'
            )

        self.add_parameter(
            'frequency',
            label='Trigger {} frequency'.format(self._trignum),
            unit='Hz',
            get_cmd=self._get_frequency,
            set_cmd=None,
            vals=vals.Numbers(0, round(self.parent._samplerate/2)),
            docstring='trigger frequency'
            )

        self.add_parameter(
            'period',
            label='Trigger {} period'.format(self._trignum),
            unit='s',
            get_cmd=self._get_period,
            set_cmd=None,
            vals=vals.Numbers(1/(self.parent._samplerate/2), 10),
            docstring='trigger period'
            )

        self.add_parameter(
            'phase',
            label='Trigger {} phase offset'.format(self._trignum),
            unit='Deg',
            get_cmd=self._get_phase_offset,
            set_cmd=None,
            vals=vals.Numbers(-180, +180),
            docstring='trigger phase offset'
            )

        self.add_parameter(
            'direction',
            label='Trigger {} direction '.format(self._trignum),
            get_cmd=self._get_direction,
            set_cmd=self._set_direction,
            vals=self._TRIGGER_DIRECTION_VALIDATION,
            docstring='trigger leading edge direction'
        )

    def _get_trigger(self):
        response = self.ask('TRIGGER:{}:SET ?'.format(self._trignum))
        trigger = response.split(',')
        pulse_state = trigger[1]
        idle_state = trigger[2]
        # trigger[0] is ACTIVE/INACTIVE
        # trigger[1,2] is the HI/LO pulse, idle state
        # trigger[3,4] is the period in seconds, ticksS
        # trigger[5,6] is the pulse width in seconds, ticks
        # trigger[7] is the number of shots
        # trigger[8,9] is the phase offset in seconds, ticks

        if response.startswith('INACTIVE'):
            active = 'OFF'
            period = 0
            period_raw = 0
            pulse_width =  0
            pulse_width_raw = 0
            num_triggers = 0
            phase_offset = 0
            phase_offset_raw = 0

        elif response.startswith('ACTIVE'):
            active = 'ON'
            period = float(trigger[3])
            period_raw = int(trigger[4])
            pulse_width =  float(trigger[5])
            pulse_width_raw = int(trigger[6])
            phase_offset = float(trigger[8])
            phase_offset_raw = int(trigger[9])
            if trigger[7] == 'INF':
                num_triggers = -1
            else:
                num_triggers = int(trigger[7])

        ret = [active,
               period, period_raw,
               pulse_width, pulse_width_raw,
               num_triggers,
               phase_offset, phase_offset_raw]

        return ret

    def _get_frequency(self):
        trigger = self._get_trigger()
        if trigger[0] == 'OFF':
            freq = 0
        else:
            freq = 1 / trigger[1]
        return freq

    def _get_period(self):
        trigger = self._get_trigger()
        if trigger[0] == 'OFF':
            per = 0
        else:
            per = trigger[1]
        return per

    def _get_period_raw(self):
        trigger = self._get_trigger()
        if trigger[0] == 'OFF':
            per = 0
        else:
            per = trigger[2]
        return per

    def _get_phase_offset(self):
        trigger = self._get_trigger()
        if trigger[0] == 'OFF':
            phase_offset = 0
        else:
            phase_offset = 360 * trigger[6] / trigger[1]
        return phase_offset

    def _set_autosync(self, autosync):
        self._autosync = autosync

    def _get_autosync(self):
        return self._autosync

    def _set_trigger(self, trigger):
        if trigger[0] == 'OFF':
            trig_cmd = 'TRIGGER:{}:SET OFF'.format(self._trignum)

        elif trigger[0] == 'ON':
            # Trigger period
            if trigger[1] != None:
                # Period has been given in seconds.
                period = trigger[1]
                period_str = '{}'.format(period)
            else:
                # Period has been given in samples.
                period_str = 'RAW{}'.format(trigger[2])
                period = trigger[2] / self.parent._samplerate
            # Trigger pulse width
            if trigger[3] != None:
                # Pulse width has been given in seconds.
                pw_str = ',{}'.format(trigger[3])
            else:
                # Pulse width has been given in samples.
                pw_str = ',RAW{}'.format(trigger[4])
            # Trigger phase offset
            if trigger[5] != None:
                # Phase offset has been given in seconds.
                phase_offs_str = ',{}'.format(trigger[5])
            elif trigger[6] != None:
                # Phase offset has been given in samples.
                phase_offs_str = ',RAW{}'.format(trigger[6])
            else:
                phase_offs_str = ''

            trig_cmd = 'TRIGGER:{}:SET '.format(self._trignum)
            trig_cmd +=  period_str + pw_str + phase_offs_str

        self.write(trig_cmd)
        if self._autosync:
            self.parent.sync()

    def _set_direction(self, direction):
        if direction == 'up':
            trig_cmd = 'TRIGGER:{}:CONFIG UP'.format(self._trignum)
        elif direction == 'down':
            trig_cmd = 'TRIGGER:{}:CONFIG DOWN'.format(self._trignum)
        else:
            self.parent._handle_error(
                'Parameter direction must be \'up\' or \'down\'')
            trig_cmd = None

        if trig_cmd != None:
            self.write(trig_cmd)

    def _get_direction(self):
        response = self.ask('TRIGGER:{}:CONFIG ?'.format(self._trignum))
        r = response.split(',')
        if (r[0] == '1') and (r[1] == '0'):
            ret = 'up'
        elif (r[0] == '0') and (r[1] == '1'):
            ret = 'down'
        elif (r[0] == '0') and (r[1] == '0'):
            warn_str = 'Trigger {} output is always LOW'.format(self._trignum)
            self.parent._handle_error(warn_str, fatal=False)
            ret = None
        elif (r[0] == '1') and (r[1] == '1'):
            warn_str = 'Trigger {} output is always HIGH'.format(self._trignum)
            self.parent._handle_error(warn_str, fatal=False)
            ret = None
        else:
            self.parent._handle_error('Invalid response received from MDAC')
            ret = None

        return ret

    def trigger_number(self):
        """ TRIG number corresponding to SMC trigger numbers"""
        return self._trignum

    def start(self, frequency, period=None, phase=None, period_raw=None):
        """
        Enables the trigger generation.

        Args:
            frequency - [Hz]
            period - [Sec]
            phase - [Deg]
            period_raw (available when period is None) - [Sample]
        """
        # Trigger period.
        if frequency != None:
            period = 1/frequency
            period_raw = None
            l_period = period
        elif period != None:
            period_raw = None
            l_period = period
        else:
            l_period = period_raw / self.parent._samplerate
        # Trigger pulse width
        if (l_period * self.parent._samplerate) < 4:
            pulse_width_raw = 1
        elif (l_period * self.parent._samplerate) < 12:
            pulse_width_raw = 2
        else:
            pulse_width_raw = 10
        # Phase
        if phase == None:
            phase_offset = 0
            phase_offset_raw = None
        elif period != None:
            phase_offset = period * phase / 360
            phase_offset_raw = None
        else:
            phase_offset = None
            phase_offset_raw = period_raw * phase / 360
        trig_cmd = ['ON',
                    period, period_raw,
                    None, pulse_width_raw,
                    phase_offset, phase_offset_raw]

        self._set_trigger(trig_cmd)

    def stop(self):
        """Disables trigger generation"""
        self._set_trigger(['OFF'])


class MDAC(VisaInstrument):
    """
    Driver for MDAC V1.x - a low noise precision voltage source
    with integrated signal routing and waveform functionality

    HW NOTES:
    don't exceed 500mA of current through any relay
    only toggle relays at zero voltage
    don't source or sink more that 20mA of current from any output
    don't connect two or more DAC outputs together

    in QT configuration, micro-D pin 13 is not connected but pin 25 is.
    DAC channel / SMC label 13 maps to micro-D #1 pin 25
    similarly, DAC channel / SMC label 45 maps to micro-D #2 pin 25

    """

    def __init__(self, name, address, baudrate=460800, debug=False,
                 logging=None, hardware_limits=True):
        """
        Instantiates the MDAC.

        Args:
            name (str): the instrument name for qcodes
            address (str): the VISA resource name

        Returns:
            MDAC object

        """
        if logging is not None:
            warnings.warn(
                'The `logging` kwarg of the MDAC init is deprecated, '
                'please use the python logging module instead.')

        self._read_buffer = collections.deque()
        self._debug = debug
        self._errors_are_fatal = not debug
        self._hardware_limits = hardware_limits

        super().__init__(name, address)

        self.visa_handle.encoding = 'latin_1'
        handle = self.visa_handle

        handle.baud_rate = baudrate
        handle.write_termination = '\r\n'
        handle.read_termination = '\n'
        handle.flow_control = 2
        handle.timeout = 50000

        _path = os.path.abspath(__file__)
        _err_path = os.path.dirname(_path) + '\\error_codes.json'

        try:
            f = open(_err_path, 'r')
            self._error_dict = json.loads(f.read())
            f.close()
        except FileNotFoundError:
            self._error_dict = {}

        # turn off Verbose output
        self.write('PE0')
        self.write('PV0')
        self.write('PLE0')

        self._read_buffer.clear()

        mdac_id = self.get_idn()
        mdac_sw_ver = float(mdac_id['firmware'])
        if(mdac_sw_ver < 1.5):
            error_str = ('Version mismatch: MDAC firmware Version {} found.\n'
                         'This Driver is for Version 1.5, \n'
                         'contact Sydney team for Firmware Update'
                         )
            self._handle_error(error_str)
        if(mdac_sw_ver > 1.7):
            error_str = ('Version mismatch: MDAC firmware Version {} found.\n'
                         'This Driver is for Version 1.5-1.7, \n'
                         'please use updated QCoDeS driver'.format(mdac_sw_ver)
                         )
            warnings.warn(error_str)

        self._samplerate = self._get_sample_rate()
        self._chan_range = self._active_channels()

        channels = ChannelList(self, "Channels", MDACChannel,
                               snapshotable=False)

        if (self._mdac_variant != 'MDAC8'):
            ch_isense = MDACExtChannel(self, 'ch_isense', 0)
            self.add_submodule('ch_isense', ch_isense)

            trig0 = Trigger(self, 'trigger0', 0)
            self.add_submodule('trigger0', trig0)

        for i in self._chan_range:
            channel = MDACChannel(self, 'chan{}'.format(i), i)
            channels.append(channel)
            self.add_submodule('ch{:02}'.format(i), channel)
        channels.lock()
        self.add_submodule('channels', channels)

        if self._hardware_limits:
            self.query('DAC:ALL:PROTECTION ON')
        else:
            self.query('DAC:ALL:PROTECTION OFF')

        self.add_parameter(
            'sample_rate',
            label='DAC update rate',
            unit='Samples/s',
            get_cmd=self._get_sample_rate,
            docstring='DAC sample update rate for all MDAC channels'
        )

        if (self._mdac_variant != 'MDAC8'):
            self.add_parameter(
                'bus',
                label='Chassis BUS relay',
                set_cmd='RELAYS:BUS_CALIBRATE {}',
                get_cmd='RELAYS:BUS_CALIBRATE ?',
                val_mapping={'open': 0, 'close': 1},
                docstring='BUS SMC RELAY\n located under the micro-D outputs'
                )

        self.add_parameter(
            'temperature',
            label='System Temperature (AVG)',
            get_cmd='STATUS:TEMPERATURE ?',
            get_parser=float,
            unit='°C',
            docstring='system average temperature\n'
                      'a number in the 50s is not unusual'
            )

        if self._hardware_limits:
            self.add_parameter(
                'protection',
                label='Output Limit Protection',
                get_cmd='DAC:ALL:PROTECTION ?',
                set_cmd='DAC:ALL:PROTECTION {}',
                get_parser=self._lowercase_parser,
                set_parser=self._uppercase_parser,
                vals=vals.OnOff(),
                docstring='globally enables / disables'
                        ' HW based limit protection'
                )

        self.add_parameter(
            'supply_voltages',
            label='MDAC supply voltages',
            get_cmd=self._supply_voltages,
            docstring='Average of system voltages \n'
                      '(5V, 3.3V, VDD, VSS)'
            )

        self.add_parameter(
            'arb_engine',
            label='MDAC AWG engine status',
            get_cmd='ARB ?',
            set_cmd='ARB {}',
            get_parser=self._lowercase_parser,
            set_parser=self._uppercase_parser,
            vals=vals.OnOff(),
            docstring='arb_engine responsible for updating dac outputs\n'
                      'if off, all waveforms stop and ramps fail to update'
            )

        self.add_function(
            'sync',
            call_cmd='DAC:ALL SYNC',
            docstring='Synchronises waveforms running on multiple channels'
            )

        self.connect_message()

    def run(self):
        """
        enables arb_engine:
        waveforms will update and ramps will continue
        """
        self.arb_engine('on')

    def stop(self):
        """
        disables arb_engine:
        all waveforms / ramps will hold at their current value
        """
        self.arb_engine('off')

    def list_triggers(self):
        """ lists the dac channels currently generating trigger outputs"""
        t = self.ask('DAC:ALL:TRIGGER ?').splitlines()
        ret = []
        for i in t:
            if i == 'ABSENT':
                ret.append(-1)
            else:
                ret.append(int(i))
        return tuple(ret)

    def _read(self):
        response = ""
        while True:
            res = self.visa_handle.read()
            res = res.replace('\r', '')
            res = res.replace('\x17', '')
            self.log.debug(f'Partial message: {res}')
            if self._debug:
                print('## <-- ' + res)
            if "[OK]" in res:
                break
            elif "[ERR:0x" in res:
                self._error_decode(res)
                break
            else:
                response += res
            response += '\n'
        resp = response.rstrip()
        self.visa_log.debug(f"Response: {resp}")
        if resp != '':
            self._read_buffer.append(resp)
        return resp

    def write(self, cmd):
        """
        write a command to the MDAC.
        checks output and puts it in a queue to be read later
        Write has to read a response because of the way the qcodes
        parameters are implemented: here setting a value gets set
        using the `write` command. Some instruments like the MDAC
        always return a status that has to be read.
        """
        if self._debug:
            print('## --> ' + cmd)
        self.visa_log.debug(f"Querying: {cmd}")
        try:
            self.visa_handle.write(cmd)
            self._read()
        except KeyboardInterrupt:
            self.clear()
            raise

    def ask(self, cmd):
        try:
            self.write(cmd)
            ret_val = self.read()
        except KeyboardInterrupt:
            self.clear()
            raise
        return ret_val

    def query(self, cmd):
        """
        write a command to the MDAC and return the response
        """
        return self.ask(cmd)

    def read(self):
        if len(self._read_buffer) == 0:
            return ''
        res = self._read_buffer.popleft()
        if "[ERR:0x" in res:
            self._error_decode(res)
            return ''

        return res

    def clear(self):
        # clears the read queue
        self._read_buffer.clear()
        self.visa_handle.clear()

    def wait_ramps(self, channels=None):
        """
        Blocks until all channels in the list have finished ramping
        'None' will wait for all channels (slow)
        """
        if channels == None:
            # a list of the current waveforms
            def get_waveforms():
                x = self._get_all_dacs_config()[1:]
                waveforms = []
                for i in x:
                    waveforms.append(i[3][0])
                return waveforms

            waveforms = get_waveforms()

            # loop until the number of ramping waveforms is small
            # so its faster to individualy check the channels
            while waveforms.count('ramp') > 11:
                waveforms = get_waveforms()

            # a list of the remaining channels
            indices = [i for i, x in enumerate(waveforms) if x == "ramp"]
            channels = []
            for i in indices:
                channels.append(self.channels[i])

        elif type(channels) == MDACChannel:
            channels = [channels]
        channels = list(channels)

        while len(channels) > 0:
            for channel in channels:
                if channel.waveform() != 'ramp':
                    channels.remove(channel)

    def _lowercase_parser(self, s):
        s = s.lower()
        return s

    def _uppercase_parser(self, s):
        s = s.upper()
        return s

    def _card_voltages(self):
        voltages = self.ask('MASTER:VOLTAGES ?').split(',')
        if voltages[0] == 'N/A':
            v5 = float(voltages[2])
        else:
            v5 = float(voltages[3])
        return v5

    def _card_voltages_raw(self):
        voltages = self.ask('MASTER:VOLTAGES ?').split(',')
        if voltages[0] == 'N/A':
            v5 = int(voltages[1],16)
        else:
            v5 = int(voltages[2],16)
        return v5

    def _supply_voltages(self):
        # Returns the system voltages
        # (5, 3.3, +12, -12)
        n = 0
        v1 = 0
        v2 = 0
        v3 = 0
        for i in range(0, len(self.channels), 8):
            r = self.channels[i]._card_voltages()
            v1 += r[0]
            v2 += r[1]
            v3 += r[2]
            n += 1
        v1 = round(v1 / n, 2)
        v2 = round(v2 / n, 2)
        v3 = round(v3 / n, 2)
        v0 = round(self._card_voltages(), 2)
        return tuple([v0, v1, v2, v3])

    def _power_good(self):
        r = self._supply_voltages()
        if r[0] < 4:  # 5V less than 4
            return False
        if r[0] > 6:  # 5V greater than 6
            return False
        if r[1] < 3:  # 3.3V less than 3
            return False
        if r[1] > 4:  # 3.3V greater than 4
            return False
        if r[2] < 8:  # +12V less than 8
            return False
        if r[2] > 16:  # +12V greater than 16
            return False
        if r[3] > -8:  # -12 more than -8
            return False
        if r[3] < -16:  # -12 lessthan -16
            return False
        return True  # all voltages within 'safe' range

    def get_idn(self):
        """generates the *IDN dictionary for qcodes"""
        reply = self.ask('?V').split(',')
        master_slave = reply[0]
        FW = reply[2]
        SN = self.ask('?H').splitlines()[0]
        SN = SN.split(',')[1].strip()
        if (float(FW) >= 1.5):
            if (master_slave == 'SLAVE'):
                self._mdac_variant = 'MDAC8'
            else:
                self._mdac_variant = self.ask('?VV')

        else:
            # pre v1.5 firmware was only ever installed on MDAC1 units
            self._mdac_variant = 'MDAC'
        id_dict = {'firmware': FW,
                   'model': 'MDAC',
                   'serial': SN,
                   'vendor': 'QNL Sydney'}
        return id_dict

    def _error_decode(self, error_string):
        # Decodes error numbers returned by the MDAC
        i0 = error_string.find('[ERR:')
        i1 = error_string.find(']')
        if not(i0 == -1 or i1 == -1):
            error_code = error_string[i0+5:i1]
        else:
            error_code = error_string

        error_str = 'ERROR: {}, ({})'.format(
            error_code, self._error_dict.get(error_code))
        self._handle_error(error_str)

    def _active_cards(self):
        cards = []
        s = self.ask('/POWER ?')
        ls = s.splitlines()
        try:
            for i in range(0, 8):
                line = ls[i].split(',')
                if 'TRUE' in line[1]:
                    cards.append(i+1)
        except (IndexError):
            pass

        return cards

    def _active_channels(self):
        cards = self._active_cards()
        channel_range_list = []
        for i in cards:
            for j in range(1, 9):
                x = ((i-1)*8)+j
                channel_range_list.append(x)
        return channel_range_list

    def _get_sample_rate(self):
        numbers = re.compile('\d+(?:\.\d+)?')
        x = self.ask('?S')
        ls = x.split(',')
        return float(ls[0])

    def _get_all_dacs_config_raw(self):
        return self.ask('DAC:ALL:CONFIG ?')

    def _get_all_dacs_config(self):
        """
        Helper function to get the active preset
        Returns a list of up to 65 items for dac_0 .. dac_64
        Each channel's list entry consists of:
          [0,1] the current output in volts, dac units.
          [2] whether the current channel has been limited.
          [3] a list describing the current waveform:
          - wave[0] is the name.  If 'off', the list stops here.
          - wave[1,2] is the period in seconds, ticks.
          - wave[3,4] is the offset/ramp_start in volts, dac units.
          - wave[5,6] is the amplitude/ramp_destination in volts, dac units.
          - wave[7] is the number of shots.
          - wave[8,9] is the phase offset in seconds, ticks.
          [4] describes the current relays (None for dac 0):
          - if not None, [main, smc, bus, term, f1, bypass, dac_output, divider]
        """
        response = self._get_all_dacs_config_raw().splitlines()

        num_lines = len(response)
        if num_lines > 0:
            all_dacs_config = []
        else:
            all_dacs_config = None

        chan_line_index = 0  # Ignore the local DAC number
        while chan_line_index < num_lines:
            # Start with output_volts, output_dac_raw, limited.
            dac_line = response[chan_line_index].split(',')
            dac_config = dac_line[1:4]

            # Add preset data.
            chan_line_index += 1
            preset_line = response[chan_line_index].split(',')
            if (preset_line[0] == 'none'):
                preset_config = ['off']
            else:
                preset_config = preset_line[0:10]
                preset_config[0] = preset_config[0].lower()
            dac_config.append(preset_config)

            # Finish with relays data.
            chan_line_index += 1
            if chan_line_index < num_lines:
                relays_line = response[chan_line_index].split(',')
                relays_config = relays_line[0:8]
                if ('TRUE' in relays_config) or ('FALSE' in relays_config):
                    # There was no relays data, this is actually data for
                    # the next channel.
                    relays_config = None
                else:
                    chan_line_index += 1
            else:
                relays_config = None

            dac_config.append(relays_config)
            all_dacs_config.append(dac_config)

        return all_dacs_config

    def _set_errors_are_fatal(self, fatal):
        self._errors_are_fatal = fatal

    def _get_errors_are_fatal(self):
        return self._errors_are_fatal

    def _handle_error(self, description):
        if self._errors_are_fatal:
            raise Exception('<<< MDAC FATAL ERROR >>> {}'.format(description))
        else:
            warnings.warn_explicit(description, UserWarning, 'MDAC', 0)
