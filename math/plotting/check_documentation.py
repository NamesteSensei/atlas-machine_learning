#!/usr/bin/env python3
"""
Script to check if a module and its functions are properly documented.
"""

import sys


def check_documentation(module_name):
    """
    Checks if the specified module and its functions have docstrings.

    Args:
        module_name (str): The name of the module to check.

    Returns:
        None
    """
    try:
        # Dynamically import the module
        module = __import__(module_name)
    except ModuleNotFoundError:
        print(f"Error: Module '{module_name}' not found.")
        sys.exit(1)

    # Check module-level docstring
    if module.__doc__ is None or len(module.__doc__.strip()) == 0:
        print(f"Module not documented or not enough: {module.__doc__} - {module}")
        sys.exit(1)

    # Check function-level docstrings
    for name, obj in vars(module).items():
        if callable(obj):
            if obj.__doc__ is None or len(obj.__doc__.strip()) == 0:
                print(f"Function '{name}' not documented or not enough: {obj.__doc__}")
                sys.exit(1)

    print("OK")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./check_documentation.py <module_name>")
        sys.exit(1)

    module_name = sys.argv[1]
    check_documentation(module_name)
