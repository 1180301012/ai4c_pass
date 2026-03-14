import sys
sys.path.insert(0, '/workspace/ai4c/samples/hf_subgraphs/fusible_subgraphs/5c/c0/5cc0a22d371768f23b67f89d644dd72cd6b92a330e367454c9587037fde226aa/pass_dir')

# Try importing both passes
try:
    import FuseLinearPermute
    print("FuseLinearPermute imported successfully")
except Exception as e:
    print(f"Error importing FuseLinearPermute: {e}")

try:
    import OptimizeTranspose
    print("OptimizeTranspose imported successfully")
except Exception as e:
    print(f"Error importing OptimizeTranspose: {e}")