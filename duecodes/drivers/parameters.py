import time
from qcodes import Parameter
from qcodes import validators as vals
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter


class CountParameter(Parameter):
    """ simple parameter that keeps track of how many times it has been called """

    def __init__(self, name: str, label: str = "Couter", **kwargs: Any):

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


# class CurrentParam1211(Parameter):
#     def __init__(self, measured_param, c_amp_ins, name="curr"):
#
#         super().__init__(name, label="ithaco1211 current", unit="A")
#
#         self._measured_param = measured_param
#         self._instrument = c_amp_ins
#
#         p_name = measured_param.name
#         p_label = getattr(measured_param, "label", None)
#         p_unit = getattr(measured_param, "unit", None)
#
#     def get_raw(self):
#         volt = self._measured_param.get()
#         current = (
#             self._instrument.sens.get() * self._instrument.sens_factor.get()
#         ) * volt
#
#         if self._instrument.invert.get():
#             current *= -1
#
#         self.cache.set(current)
#         return current
