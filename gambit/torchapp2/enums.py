from enum import Enum
from torch import nn


class ActivationError(Exception):
    """An exception used in the TorchApp Activation module."""

    pass


class Activation(str, Enum):
    """
    Non-linear activation functions used in pytorch

    See https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    Excludes activation funtions that require arguments (i.e. MultiheadAttention and Threshold).
    """

    ELU = "ELU"
    Hardshrink = "Hardshrink"
    Hardsigmoid = "Hardsigmoid"
    Hardtanh = "Hardtanh"
    Hardswish = "Hardswish"
    LeakyReLU = "LeakyReLU"
    LogSigmoid = "LogSigmoid"
    PReLU = "PReLU"
    ReLU = "ReLU"
    ReLU6 = "ReLU6"
    RReLU = "RReLU"
    SELU = "SELU"
    CELU = "CELU"
    GELU = "GELU"
    Sigmoid = "Sigmoid"
    SiLU = "SiLU"
    Mish = "Mish"
    Softplus = "Softplus"
    Softshrink = "Softshrink"
    Softsign = "Softsign"
    Tanh = "Tanh"
    Tanhshrink = "Tanhshrink"
    GLU = "GLU"

    def __str__(self):
        return self.value

    def module(self, *args, **kwargs):
        """
        Returns the pytorch module for this activation function.

        Args:
            args: Arguments to pass to the function to create the module.
            kwargs: Keyword arguments to pass to the function to create the module.
        Raises:
            ActivationError: If the activation function is not available in pytorch

        Returns:
            nn.Module: The pytorch module for this activation function.
        """
        if not hasattr(nn, self.value):
            raise ActivationError(f"Activation function '{self.value}' not available.")

        return getattr(nn, self.value)(*args, **kwargs)

    @classmethod
    def default_tune_choices(cls):
        return [
            cls.ReLU,
            cls.LeakyReLU,
            cls.RReLU,
            cls.ELU,
            cls.Hardsigmoid,
        ]