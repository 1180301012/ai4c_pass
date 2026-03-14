import torch
import triton
import triton.language as tl

def pattern(x, batch_size):
    # Dynamic reshape + permute pattern that works for different batch sizes
    # This should match: x.reshape(batch_size, 16, 12, -1).permute(0, 3, 1, 2)
    out = x.reshape(batch_size, 16, 12, -1).permute(0, 3, 1, 2)
    return out

def replacement_args(x, batch_size):
    return (x, batch_size)

@torch.fx.wrap  
def debug_reshape(x, batch_size):
    """Debug reshape operation"""
    n_batch, seq_len, n_features = x.shape
    height, width = 16, 12
    
    print(f"Debug: Input shape: {x.shape}")
    print(f"Debug: batch_size parameter: {batch_size}")
    print(f"Debug: seq_len // (height * width) = {seq_len} // ({height} * {width}) = {seq_len // (height * width)}")
    
    # Use PyTorch for now to check correctness
    out = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
    print(f"Debug: Output shape: {out.shape}")
    
    return out

def replacement_func():
    return debug_reshape