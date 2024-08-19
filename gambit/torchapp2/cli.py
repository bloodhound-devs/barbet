from typing import Callable
import typer
from inspect import signature, Parameter
from functools import wraps
from dataclasses import dataclass

@dataclass
class Method():
    func: Callable
    methods_to_call: list[str]
    main: bool = False    
    tool: bool = False    
    signature_ready: bool = False
    obj = None

    @property
    def __name__(self):
        return self.func.__name__

    def __call__(self, *args, **kwargs):
        func_args = {k: v for k, v in kwargs.items() if k in signature(self.func).parameters}
        return self.func(self.obj, *args, **func_args)

    @property
    def __signature__(self):
        return signature(self.func)

    @property
    def __doc__(self):
        return self.func.__doc__


def method(*args, main:bool=False, tool:bool=False):
    if len(args) == 1 and callable(args[0]):
        return Method(args[0], [], main=main, tool=tool)
    
    def decorator(func):
        return Method(func, args, main=main, tool=tool)

    return decorator


def tool(*methods_to_call):
    return method(*methods_to_call, tool=True)


def main(*methods_to_call):
    return method(*methods_to_call, main=True, tool=True)


def collect_arguments(*funcs):
    """Collect arguments from multiple functions."""
    params = {}
    for func in funcs:
        for name, param in signature(func).parameters.items():
            # if param.default != Parameter.empty:
            #     # only include args with defaults
            #     continue
            if name != "self":  # Exclude 'self' parameter
                params[name] = param
    return params


class CLIApp:
    def __init__(self):
        self.main_app = typer.Typer()
        self.tools_app = typer.Typer()
        self.register_methods()

    @classmethod
    def main(cls):
        cls().main_app()

    @classmethod
    def tools(cls):
        cls().tools_app()

    def add_to_main(self, func):
        self.main_app.command()(func)
        return func

    def add_to_tools(self, func):
        self.tools_app.command()(func)
        return func

    def register_methods(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            if not isinstance(attr, Method):
                continue

            # Add to the CLI if method is decorated as a command
            if attr.main:
                self.add_to_main(attr)
            if attr.tool:
                self.add_to_tools(attr)

            # Modify the signature of the method if necessary
            if not attr.signature_ready:
                self.modify_signature(attr)

    def modify_signature(self, method_to_modify:Method, **kwargs) -> None:
        # Check if the method is already had its signature modified
        if not isinstance(method_to_modify, Method) or method_to_modify.signature_ready:
            return

        method_to_modify.obj = self

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
            try:
                method_to_modify.func.__signature__ = signature(method_to_modify.func).replace(parameters=new_params)
            except Exception:
                print(f"ERROR in {method_to_modify}")
        
        # Set the method as ready
        method_to_modify.signature_ready = True
