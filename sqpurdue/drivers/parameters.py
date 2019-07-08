import time
import qcodes as qc

class CounterParam(qc.Parameter):
    def __init__(self, name):
        # only name is required
        super().__init__(name, label='Times this has been read',
                         vals=qc.validators.Ints(min_value=0),
                         docstring='counts how many times get has been called '
                                   'but can be reset to any integer >= 0 by set')
        self._count = 0

    def get_raw(self):
        out = self._count
        self._count += 1
        return out

    def set_raw(self, val):
        self._count = val

class TimerParam(qc.Parameter):

    def __init__(self, name):
        # only name is required
        super().__init__(name, label='time elapsed', unit='s',
                         docstring='number of seconds elapsed from the \
                                    last call to TimerParam.__init__ or \
                                    TimerParam.reset()')

        self.tstart = time.time()

    def reset(self):
        self.tstart = time.time()

    def get_raw(self):
        return time.time() - self.tstart

class CurrentParam1211(qc.Parameter):

    def __init__(self, measured_param, c_amp_ins, name='curr'):


        super().__init__(name, label='ithaco1211 current', unit='A')

        self._measured_param = measured_param
        self._instrument = c_amp_ins

        p_name = measured_param.name
        p_label = getattr(measured_param, 'label', None)
        p_unit = getattr(measured_param, 'unit', None)

    def get_raw(self):
        volt = self._measured_param.get()
        current = (self._instrument.sens.get() *
                   self._instrument.sens_factor.get()) * volt

        if self._instrument.invert.get():
            current *= -1

        value = current
        self._save_val(value)
        return value

class CondParam2probe(qc.Parameter):

    def __init__(self, current_param, voltage_param, name='cond'):


        super().__init__(name, label='ac conductance', unit='Conductance')

    #     self._measured_param = measured_param
    #     self._instrument = c_amp_ins
    #
    #     p_name = measured_param.name
    #     p_label = getattr(measured_param, 'label', None)
    #     p_unit = getattr(measured_param, 'unit', None)
    #
    # def get_raw(self):
    #     volt = self._measured_param.get()
    #     current = (self._instrument.sens.get() *
    #                self._instrument.sens_factor.get()) * volt
    #
    #     if self._instrument.invert.get():
    #         current *= -1
    #
    #     value = current
    #     self._save_val(value)
    #     return value
