import time
from qcodes import Parameter
from qcodes import validators as vals
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter


class CountParameter(Parameter):
    """ simple parameter that keeps track of how many times it has been called """

    def __init__(self, name: str, label: str = "Couter", **kwargs):

        hardcoded_kwargs = ["unit", "get_cmd", "set_cmd"]

        for hck in hardcoded_kwargs:
            if hck in kwargs:
                raise ValueError(f'Can not set "{hck}" for an ' "CountParameter.")

        super().__init__(
            name,
            label=label,
            unit="#",
            vals=vals.Ints(min_value=0),
            set_cmd=False,
            **kwargs,
        )
        self._count = 0

    def get_raw(self):
        out = self._count
        self._count += 1
        return out

    def reset_count(self) -> None:
        self._count = 0


class CurrentParam1211(Parameter):
    def __init__(self, measured_param, c_amp_ins, name="curr"):

        super().__init__(name, label="ithaco1211 current", unit="A")

        self._measured_param = measured_param
        self._instrument = c_amp_ins

        p_name = measured_param.name
        p_label = getattr(measured_param, "label", None)
        p_unit = getattr(measured_param, "unit", None)

    def get_raw(self):
        volt = self._measured_param.get()
        current = (
            self._instrument.sens.get() * self._instrument.sens_factor.get()
        ) * volt

        if self._instrument.invert.get():
            current *= -1

        self.cache.set(current)
        return current

class AutoRangedSRSVoltage(Parameter):

    def __init__(self, complex_param, max_changes=1):
        
        self._complex_param = complex_param
        self._parent_lockin = self._complex_param.instrument
        self.max_changes = max_changes
        
        param_name = self._complex_param.name
        param_label = getattr(self._complex_param, "label", None)
        param_unit = getattr(self._complex_param, "unit", None)

        super().__init__(
            param_name+'_auto', label=param_label, unit=param_unit
        )

    def _increment_sens(self, current_sens, dir: str):

        n_to = self._parent_lockin._N_TO_VOLT
        to_n = self._parent_lockin._VOLT_TO_N

        n = to_n[current_sens]
        rel_sens = dict((k, v - current_sens) for k,v in n_to.items())
        del rel_sens[n]

        if dir=='up':
            smallest_rv = 2.0
            for k, rv in rel_sens.items():
                if (rv > 0.0) and (rv < smallest_rv):
                    smallest_rv = rv
                    m = k
        elif dir=='down':
            largest_rv = -2.0
            for k, rv in rel_sens.items():
                if (rv < 0.0) and (rv > largest_rv):
                    largest_rv = rv
                    m = k

        if m > max(n_to.keys()) or m < min(n_to.keys()):
            return None
        else:
            self._parent_lockin.sensitivity.set(n_to[m])
            return n_to[m]
        
    def get_raw(self):

        time_const = self._parent_lockin.time_constant.get_latest()
        val = self._complex_param.get()
        sens = self._parent_lockin.sensitivity.get()

        for i in range(self.max_changes):
            
            if abs(val) > 0.8 * sens:
                # range is too small
                sens = self._increment_sens(sens, 'up')
            elif abs(val) < 0.1 * sens:
                # range is too large
                sens = self._increment_sens(sens, 'down')
            else:
                break

            if sens is None:
                break
            else:
                time.sleep(time_const)
                val = self._complex_param.get()

        else:
            return val