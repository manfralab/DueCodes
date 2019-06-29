# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:12:10 2019

@author: Teng
"""
import logging
from qcodes import VisaInstrument
from qcodes import validators as vals

#### I/B (A/T) ratio
ratio = 12.371
#### Maximum B-field
B_max = 5.004

log = logging.getLogger(__name__)

class Kepco_BOP10_100GL(VisaInstrument):

     def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)
        self.add_parameter(name='current',
                           label='output current',
                           get_cmd='CURR?',
                           set_cmd='CURR {}',
                           unit='A',
                           get_parser=float,
                           docstring="Read/Set current (A)")

        self.add_parameter(name='current_limit',
                           label='current limit',
                           get_cmd='CURR:LIM?',
                           set_cmd='CURR:PROT:LIM {}',
                           unit='A',
                           docstring="Read/Set current limit (A)")

        # self.add_parameter(name='Field',
        #                    label='B_field',
        #                    unit='T',
        #                    get_cmd='CURR?',
        #                    set_cmd='CURR {}',
        #                    vals=vals.Numbers(B_max*-1, B_max),
        #                    get_parser=self._get_B_field,
        #                    set_parser=self._set_B_field,
        #                    docstring="Read/Set B-field in T")

        self.add_parameter(name='voltage',
                           label='output voltage',
                           unit='V',
                           get_cmd='MEAS:VOLT?',
                           get_parser=float,
                           docstring="Read output voltage (V)")

     # def _get_B_field(self,s):
     #    log.info('Get B-field (T)')
     #    fld = float(s)/ratio
     #    return fld # convert current(A) into B field (T)
     #
     #
     # def _set_B_field(self,s):
     #     log.info('Set B-field (T)')
     #     cmd = round(float(s)*ratio,8)
     #     return cmd
