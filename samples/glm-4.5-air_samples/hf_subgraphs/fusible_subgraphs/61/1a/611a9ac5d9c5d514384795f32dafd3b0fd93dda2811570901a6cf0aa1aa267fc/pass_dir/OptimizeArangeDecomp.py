import torch
import triton
import triton.language as tl

def pattern():
    # Create a simple pattern that matches constant tensor creation
    # This should be very simple and avoid dead code issues
    result = torch.arange(1)
    return result,

def replacement_args():
    return ()

@torch.fx.wrap
def create_const_tensor():
    # Create constant tensor with value 1 using safe operations
    out = torch.empty((), dtype=torch.float32)
    out.fill_(1.0)
    return out

def replacement_func():
    return create_const_tensor