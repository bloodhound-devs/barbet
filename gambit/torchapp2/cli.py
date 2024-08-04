import typer
from inspect import signature, Parameter
from functools import wraps

def cli(func):
    func._is_cli_command = True
    return func


def collect_arguments(*funcs):
    """Collect arguments from multiple functions."""
    params = {}
    for func in funcs:
        for name, param in signature(func).parameters.items():
            if name != "self":  # Exclude 'self' parameter
                params[name] = param
    return params


def calls(*called_funcs):
    """Decorator to register that a method calls other methods."""
    def decorator(func):
        breakpoint()
        @wraps(func)
        def wrapper(self, **kwargs):
            for called_func in called_funcs:
                called_args = {k: v for k, v in kwargs.items() if k in signature(called_func).parameters}
                called_func(self, **called_args)
            func(self, **kwargs)
        wrapper._is_cli_command = True
        # Collect arguments from called functions
        params = collect_arguments(*called_funcs)
        # Include parameters from the main function itself
        func_params = {name: param for name, param in signature(func).parameters.items() if name != "self" and name != "kwargs"}
        params.update(func_params)
        new_params = [
            Parameter(name, param.kind, default=param.default, annotation=param.annotation)
            for name, param in params.items()
        ]
        wrapper.__signature__ = signature(func).replace(parameters=new_params)
        return wrapper
    return decorator


def call(func, **kwargs):
    """Helper function to call a function with filtered arguments."""
    func_args = {k: v for k, v in kwargs.items() if k in signature(func).parameters}
    return func(**func_args)


class CLIApp:
    def __init__(self):
        self.app = typer.Typer()
        self.register_commands()

    def main(self):
        self.app()

    def add_command(self, func):
        self.app.command()(func)
        return func

    def register_commands(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_is_cli_command"):
                self.add_command(attr)


