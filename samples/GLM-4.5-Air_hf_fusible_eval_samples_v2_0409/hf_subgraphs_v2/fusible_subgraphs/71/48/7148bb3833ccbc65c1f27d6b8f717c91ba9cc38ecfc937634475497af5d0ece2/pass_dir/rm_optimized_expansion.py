import os

# Remove the separate optimized expansion pass since it's now included
# in the fused feature processing pipeline
if os.path.exists('./pass_dir/OptimizedScaleExpansion.py'):
    os.remove('./pass_dir/OptimizedScaleExpansion.py')
    print("Removed OptimizedScaleExpansion.py")
else:
    print("OptimizedScaleExpansion.py not found")