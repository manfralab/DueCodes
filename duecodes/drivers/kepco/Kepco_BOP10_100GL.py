# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:12:10 2019

@author: Teng
"""
# import logging
from qcodes import VisaInstrument
from qcodes import validators as val

# LOGGER = logging.getLogger(__name__)


class Kepco_BOP10_100GL(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\n", **kwargs)

        self.add_parameter(
            name="current",
            label="output current",
            get_cmd=self._get_current,
            set_cmd=self._set_current,
            unit="A",
            vals=val.Numbers(-65, 65),
            docstring="get/set current (A)",
        )

        self.add_parameter(
            name="current_limit",
            label="current limit",
            get_cmd="CURR:LIM?",
            set_cmd="CURR:PROT:LIM {}",
            get_parser=float,
            set_parser=float,
            unit="A",
            docstring="get/set current limit (A)",
        )

        self.add_parameter(
            name="voltage",
            label="output voltage",
            unit="V",
            get_cmd="MEAS:VOLT?",
            get_parser=float,
            docstring="get output voltage (V)",
        )

    def _set_current(self, value):
        command = f"CURR {value:.5f}"
        self.write(command)

    def _get_current(self):
        return float(self.ask("CURR?"))
