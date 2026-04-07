import sys
sys.path.append('./pass_dir')

try:
    import FoldViewExpandToBroadcast_1_2_64_8_8
    print("✓ Successfully imported FoldViewExpandToBroadcast_1_2_64_8_8 module")
    from FoldViewExpandToBroadcast_1_2_64_8_8 import pattern as pat2, replacement_args as args2, replacement_func as func2
    print("Pattern function:", pat2)
    print("Args function:", args2)
    print("Function:", func2())
except Exception as e:
    print("✗ Failed to import FoldViewExpandToBroadcast_1_2_64_8_8:", e)
    import traceback
    traceback.print_exc()

try:
    import NormalizeAlongDim2
    print("\n✓ Successfully imported NormalizeAlongDim2 module")
    from NormalizeAlongDim2 import pattern as pat1, replacement_args as args1, replacement_func as func1
    print("Pattern function:", pat1)
    print("Args function:", args1)
    print("Function:", func1())
except Exception as e:
    print("\n✗ Failed to import NormalizeAlongDim2:", e)
    import traceback
    traceback.print_exc()