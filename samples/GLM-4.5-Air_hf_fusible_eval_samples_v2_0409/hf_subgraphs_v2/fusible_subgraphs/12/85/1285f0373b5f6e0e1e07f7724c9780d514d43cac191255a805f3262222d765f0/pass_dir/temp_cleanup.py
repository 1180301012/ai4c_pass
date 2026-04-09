# This file helps cleanup - we'll just keep the properly named files
import os

# Remove the old file
if os.path.exists('./pass_dir/OptimizeGraphIndexConcatOnes.py'):
    os.remove('./pass_dir/OptimizeGraphIndexConcatOnes.py')
    print("Removed old file")

print("Files in pass_dir:")
for f in os.listdir('./pass_dir'):
    if f.endswith('.py'):
        print(f"  {f}")