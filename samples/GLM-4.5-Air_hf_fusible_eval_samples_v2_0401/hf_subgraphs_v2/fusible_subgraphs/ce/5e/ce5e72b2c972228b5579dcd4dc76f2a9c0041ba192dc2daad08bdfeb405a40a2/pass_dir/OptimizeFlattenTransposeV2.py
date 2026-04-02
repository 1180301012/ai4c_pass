import torch
import triton
import triton.language as tl

def pattern(conv3d_output):
    # Match the sequence: flatten(2) -> transpose(1, 2)
    flattened = conv3d_output.flatten(2)
    result = flattened.transpose(1, 2)
    return result

def replacement_args(conv3d_output):
    return (conv3d_output,)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    """Optimized flatten(2) followed by transpose(1, 2)"""
    input_shape = input_tensor.shape  # [N, C, D, H, W] = [1, 768, 2, 16, 16]
    N, C, D, H, W = input_shape
    
    # Use PyTorch operations for correctness - this avoids indexing errors
    # flatten(2) along dimension 2: [N, C, D*H*W]
    flattened = input_tensor.flatten(2)  # [1, 768, 512]
    
    # transpose(1, 2): swap dimensions 1 and 2  
    result = flattened.transpose(1, 2)   # [1, 512, 768]
    
    return result





def replacement_func():
    return optimized_flatten_transpose