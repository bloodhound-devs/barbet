from typing import Callable
import typer
from inspect import signature, Parameter
from functools import wraps
from dataclasses import dataclass

@dataclass
class Method():
    func: Callable
    methods_to_call: list[str]
    command: bool = False    
    signature_ready: bool = False

    @property
    def __name__(self):
        return self.func.__name__

    def __call__(self, *args, **kwargs):
        func_args = {k: v for k, v in kwargs.items() if k in signature(self.func).parameters}
        return self.func(*args, **func_args)

    @property
    def __signature__(self):
        return signature(self.func)


def method(*args, command:bool=False):
    if len(args) == 1 and callable(args[0]):
        return Method(args[0], [], command)
    
    def decorator(func):
        return Method(func, args, command)

    return decorator


def command(*methods_to_call):
    return method(*methods_to_call, command=True)


def collect_arguments(*funcs):
    """Collect arguments from multiple functions."""
    params = {}
    for func in funcs:
        for name, param in signature(func).parameters.items():
            if name != "self":  # Exclude 'self' parameter
                params[name] = param
    return params


class CLIApp:
    def __init__(self):
        self.app = typer.Typer()
        self.register_methods()

    @classmethod
    def main(cls):
        cls().app()

    def add_command(self, func):
        self.app.command()(func)
        return func

    def register_methods(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            if not isinstance(attr, Method):
                continue

            # Add to the CLI if method is decorated as a command
            if attr.command:
                self.add_command(attr)

            # Modify the signature of the method if necessary
            if not attr.signature_ready:
                self.modify_signature(attr)


    def modify_signature(self, method_to_modify:Method, **kwargs) -> None:
        # Check if the method is already had its signature modified
        if not isinstance(method_to_modify, Method) or method_to_modify.signature_ready:
            return

        all_methods = [method_to_modify]
        for method_to_call_name in method_to_modify.methods_to_call:
            method_to_call = getattr(self, method_to_call_name)

            # make sure method is has its signature modified before getting parameters
            self.modify_signature(method_to_call)
            all_methods.append(method_to_call)

        # Get all arguments from all methods
        params = collect_arguments(*all_methods)
        new_params = [
            Parameter(name, param.kind, default=param.default, annotation=param.annotation)
            for name, param in params.items()
            if name not in ["self", "kwargs"]
        ]
        if new_params:        
            method_to_modify.func.__signature__ = signature(method_to_modify.func).replace(parameters=new_params)
        
        # Set the method as ready
        method_to_modify.signature_ready = True
