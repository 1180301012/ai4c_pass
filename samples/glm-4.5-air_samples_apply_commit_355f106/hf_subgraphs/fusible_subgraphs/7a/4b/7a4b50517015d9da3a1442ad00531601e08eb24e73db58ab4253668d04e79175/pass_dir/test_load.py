# Test script to check if both passes can be loaded
import sys
import os
sys.path.append('./pass_dir')

try:
    import RemoveDropoutRate0
    print("✅ RemoveDropoutRate0 loaded successfully")
except Exception as e:
    print(f"❌ RemoveDropoutRate0 failed to load: {e}")

try:
    import FuseSiLUWithMultiply
    print("✅ FuseSiLUWithMultiply loaded successfully")
except Exception as e:
    print(f"❌ FuseSiLUWithMultiply failed to load: {e}")

# Check JSON file
import json
try:
    with open('./pass_dir/sorted_output_pass_rule_names.json', 'r') as f:
        config = json.load(f)
        print(f"✅ JSON config loaded: {config}")
        
        # Check if all passes in config exist as files
        for pass_name in config:
            if f"{pass_name}.py" not in os.listdir('./pass_dir'):
                print(f"❌ Pass file missing: {pass_name}.py")
            else:
                print(f"✅ Pass file found: {pass_name}.py")
                
except Exception as e:
    print(f"❌ JSON config failed to load: {e}")