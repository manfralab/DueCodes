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

    # should work with X, Y or complex parameters from SR830 or SR860 lock ins
    
    def __init__(self, voltage_param, max_changes=1):
        
        self._voltage_param = voltage_param

        self._parent_lockin = self._voltage_param.instrument
        self._n_to = self._parent_lockin._N_TO_VOLT
        self._to_n = self._parent_lockin._VOLT_TO_N

        self.max_changes = max_changes
        
        param_name = self._voltage_param.name
        param_label = getattr(self._voltage_param, "label", None)
        param_unit = getattr(self._voltage_param, "unit", None)
        param_vals = getattr(self._voltage_param, "vals", None)

        super().__init__(
            param_name+'_auto', 
            label=param_label, unit=param_unit, vals=param_vals,
        )

    def _increment_sens(self, current_sens, dir: str):
        # this was needed thanks to some inconsistencies in the
        # SR830 and SR86X drivers

        n = self._to_n[current_sens]
        m = -1

        if dir=='up':
            smallest_rv = 2.0
            for k, v in self._n_to.items():
                if k!=n:
                    rv = v - current_sens
                    if (rv > 0.0) and (rv < smallest_rv):
                        smallest_rv = rv
                        m = k
        elif dir=='down':
            largest_rv = -2.0
            for k, v in self._n_to.items():
                if k!=n:
                    rv = v - current_sens
                    if (rv < 0.0) and (rv > largest_rv):
                        largest_rv = rv
                        m = k

        if m == -1:
            return None
        else:
            self._parent_lockin.sensitivity.set(self._n_to[m])
            return self._n_to[m]
        
    def _time_constant_wait(self):
        # wait for one time constant
        # substract the time it took to get the time constant 
        # to avoid waiting longer than necessary
        tstart = time.time()
        time_constant = self._parent_lockin.time_constant.get()
        while ( (time.time() - tstart) < time_constant):
            time.sleep(1e-3)

    def get_raw(self):

        val = self._voltage_param.get()
        sens = self._parent_lockin.sensitivity.get()

        for i in range(self.max_changes):
            
            if abs(val) > 0.8 * sens:
                # range is too small
                sens = self._increment_sens(sens, 'up')
            elif abs(val) < 0.1 * sens:
                # range is too large
                sens = self._increment_sens(sens, 'down')
            else:
                # no change needed
                break

            if sens is None:
                # hit range limit
                break
            else:
                self._time_constant_wait()                
                val = self._voltage_param.get()

        return val