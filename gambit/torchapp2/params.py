from typer.models import OptionInfo
from .enums import Activation


class Param(OptionInfo):
    def __init__(
        self,
        default=None,
        tune=False,
        tune_min=None,
        tune_max=None,
        tune_choices=None,
        log=False, # deprecated. use tune_log
        tune_log=False,
        distribution=None,
        annotation=None,
        **kwargs,
    ):
        super().__init__(default=default, **kwargs)
        self.tune = tune
        self.tune_log = tune_log or log
        self.tune_min = tune_min if tune_min is not None else self.min
        self.tune_max = tune_max if tune_max is not None else self.max
        self.annotation = annotation
        self.distribution = distribution
        self.tune_choices = tune_choices
        
        if distribution:
            raise NotImplementedError("Distribution for parameters not implemented yet")

    def check_choices(self):
        if self.tune_choices:
            return

        if self.annotation == bool:
            self.tune_choices = [False, True]

        elif self.annotation == Activation:
            self.tune_choices = Activation.default_tune_choices()
