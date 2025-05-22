#!/usr/bin/env python3
"""
Basic test script to verify our changes to tools.py and site_agents.py
This script only checks if the files exist and can be parsed by Python
"""
import os
import sys
import ast

def check_file_syntax(file_path):
    """Check if a Python file has valid syntax"""
    print(f"Checking syntax for {file_path}...")
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✅ {file_path} has valid syntax")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")
        return False

if __name__ == "__main__":
    print("Starting basic syntax tests...")
    
    # Define the files to check
    files_to_check = [
        "app/tools.py",
        "app/site_agents.py",
        "app/schemas.py"
    ]
    
    # Check each file
    success = True
    for file_path in files_to_check:
        if not check_file_syntax(file_path):
            success = False
    
    if success:
        print("🎉 All files have valid syntax!")
        sys.exit(0)
    else:
        print("❌ Some files have syntax errors")
        sys.exit(1)
