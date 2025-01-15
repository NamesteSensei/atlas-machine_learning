#!/usr/bin/env python3
"""
This script checks if a Python module is properly documented.
"""

import ast
import sys

def check_module_documentation(file_path):
    """
    Check if the module and its functions are properly documented.

    Args:
        file_path (str): Path to the Python file to be checked.

    Returns:
        bool: True if the module is documented properly, False otherwise.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        parsed = ast.parse(content)
    except SyntaxError as e:
        print(f"SyntaxError in {file_path}: {e}")
        return False

    # Check for module-level docstring
    if not ast.get_docstring(parsed):
        print(f"Module not documented or not enough: {file_path}")
        return False

    # Check for function-level docstrings
    for node in ast.walk(parsed):
        if isinstance(node, ast.FunctionDef):
            if not ast.get_docstring(node):
                print(f"Function '{node.name}' is not documented in {file_path}")
                return False

    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./check_documentation.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if check_module_documentation(file_path):
        print("OK")
        sys.exit(0)
    else:
        sys.exit(1)
