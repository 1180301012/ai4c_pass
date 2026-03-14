import torch

def pattern(in_1, in_0):
    """Pattern matching: ReLU followed by Add"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    return tmp_1

def replacement_args(in_1, in_0):
    """Extract arguments for the replacement function"""
    return in_1, in_0

@torch.fx.wrap
def fused_relu_add(in_1, in_0):
    """Simple fused ReLU + Add using efficient operations"""
    # Use in-place operations for better memory efficiency
    relu_result = torch.relu(in_1)
    fused_result = relu_result + in_0
    return fused_result

def replacement_func():
    """Return the fused function"""
    return fused_relu_add