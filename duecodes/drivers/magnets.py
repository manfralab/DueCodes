# -*- coding: utf-8 -*-
"""
Specific drivers for magnets throughout the lab. Should subclass instrument
or use multiple instruments used as controllers and contain magnet
calibration information as attributes
"""

# import logging
from warnings import warn
from qcodes import validators as vals
from duecodes.drivers.kepco.Kepco_BOP10_100GL import Kepco_BOP10_100GL

# LOGGER = logging.getLogger(__name__)


class Oxford5TMagnet(Kepco_BOP10_100GL):
    """
    Driver for the Oxford 4K/5T magnet dewar sitting in the middle of B60.
    This driver assumes you are using the Kepco BOP10_100GL power supply
    """

    def __init__(self, name, address, reset=False, **kwargs):

        self.amp_per_tesla = 12.371  # A/T
        self.max_field = 5.00  # T
        self.min_field_step = 0.001  # T
        self._ramp_rate = 100

        super().__init__(name, address, **kwargs)

        self.current.step = self.min_field_step * self.amp_per_tesla

        self.add_parameter(
            name="field",
            label="B_field",
            unit="T",
            get_cmd=self._get_field,
            set_cmd=self._set_field,
            vals=vals.Numbers(-1 * self.max_field, self.max_field),
            step=self.min_field_step,
            docstring="get/set B-field in T",
        )

        self.add_parameter(
            name="ramp_rate",
            label="ramp_rate",
            unit="mT/min",
            set_cmd=self._set_ramp_rate,
            get_cmd=self._get_ramp_rate,
            vals=vals.Numbers(0, 200),
            initial_value=self._ramp_rate,
            docstring="get/set ramp_rate in mT/s",
        )

        self.connect_message()

    def connect_message(self):
        idn = self.IDN()
        con_msg = ('Connected to: Oxford 5T magnet. Using {vendor} {model} power supply '
                   '(serial:{serial}, firmware:{firmware})'.format(**idn))
        print(con_msg)
        warn(
            "Switch heater on magnet must be set with the attached DC power supply."
        )

    def _get_field(self):
        # LOGGER.info("get field (T)")
        current = self.current.get()
        field = current / self.amp_per_tesla
        return round(field, 6)

    def _set_field(self, field):
        # LOGGER.info("set field (T)")
        set_point = round(field * self.amp_per_tesla, 6)
        self.current.set(set_point)

    def _set_ramp_rate(self, ramp_rate):
        # set ramp rates for field
        self._ramp_rate = ramp_rate
        tesla_per_sec = ramp_rate * 1e-3 / 60.0
        amp_per_sec = tesla_per_sec * self.amp_per_tesla
        self.current.inter_delay = self.current.step / amp_per_sec

    def _get_ramp_rate(self):
        return self._ramp_rate
