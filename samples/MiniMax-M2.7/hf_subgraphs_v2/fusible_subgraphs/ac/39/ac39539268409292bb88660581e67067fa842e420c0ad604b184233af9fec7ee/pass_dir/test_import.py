import sys
sys.path.insert(0, './pass_dir')
try:
    import PositionEncodingFusion
    print("Import successful!")
    print("Functions:", dir(PositionEncodingFusion))
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()