# -*- coding: utf-8 -*-
"""
Driver for the Oxford 4K/5T magnet dewar sitting in the middle of B60

This driver assumes you are using the Kepco BOP10_100GL power supply
"""
import logging
from qcodes import VisaInstrument
from qcodes import validators as val
from .Kepco.Kepco_BOP10_100GL import Kepco_BOP10_100GL

AMP_PER_TESLA = 12.371 # A/T
MAX_FIELD = 5.004 # T

log = logging.getLogger(__name__)

class Oxford5TMagnet(Kepco_BOP10_100GL):

     def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(self, name, address, reset=reset, **kwargs)

        self.add_parameter(name='Field',
                           label='B_field',
                           unit='T',
                           get_cmd=self._get_field,
                           set_cmd=self._set_field,
                           vals=vals.Numbers(-1*MAX_FIELD, MAX_FIELD),
                           docstring="Read/Set B-field in T")

     def _get_field(self):
        log.info('get field (T)')
        current = self.current.get()
        field = current/AMP_PER_TESLA
        return field

     def _set_field(self,field):
         log.info('set field (T)')
         setpoint = round(field*AMP_PER_TESLA,8)
