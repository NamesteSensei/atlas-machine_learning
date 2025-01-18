#!/usr/bin/env python3
"""
Script to check documentation for a Python module.

Ensures that the provided module, its classes, and methods/functions have
appropriate docstrings, meeting documentation requirements.
"""

import sys
import inspect
import importlib


def check_docstring(obj, name):
    """
    Check if an object (module, class, or function) has a docstring.

    Args:
        obj: The object to check.
        name: The name of the object being checked (for reporting).

    Returns:
        bool: True if the object has a docstring, False otherwise.
    """
    if not inspect.getdoc(obj):
        print(f"Missing docstring for: {name}")
        return False
    return True


def check_module(module_name):
    """
    Check documentation for a Python module, including its classes and methods.

    Args:
        module_name: The name of the module to check.

    Returns:
        bool: True if all required docstrings are present, False otherwise.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error importing module '{module_name}': {e}")
        return False

    print(f"Checking documentation for module: {module_name}")
    all_documented = check_docstring(module, f"Module '{module_name}'")

    # Check documentation for all classes in the module
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module_name:  # Ensure the class is from this module
            all_documented &= check_docstring(cls, f"Class '{name}'")

            # Check documentation for all methods in the class
            for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                all_documented &= check_docstring(
                    method, f"Method '{cls.__name__}.{method_name}'"
                )

    # Check documentation for standalone functions in the module
    for name, func in inspect.getmembers(module, inspect.isfunction):
        all_documented &= check_docstring(func, f"Function '{name}'")

    return all_documented


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./check_documentation.py <module_name>")
        sys.exit(1)

    module_name = sys.argv[1]
    if check_module(module_name):
        print("OK")
        sys.exit(0)
    else:
        print("Module not documented or not enough:")
        sys.exit(1)
