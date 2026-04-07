#!/usr/bin/env python3
import os
import shutil

# Remove unused pass files that don't match computation patterns
unused_files = [
    'FuseElementWiseOperations_300x1x256.py',
    'FuseDoubleSigmoidOperations.py', 
    'TestSimplePass.py'
]

for file in unused_files:
    file_path = f'./pass_dir/{file}'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed: {file_path}")

# Remove __pycache__ directory
cache_dir = './pass_dir/__pycache__'
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Removed: {cache_dir}")

print("Cleaned up unused pass files successfully!")