import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern: linear transformation followed by dropout with rate=0.0, then transpose"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)  # rate=0.0 makes this a no-op
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for replacement - all inputs are needed"""
    return (in_0, in_1, in_2)

@torch.fx.wrap
def eliminate_zero_dropout(in_0, in_1, in_2):
    """
    Optimized implementation that eliminates zero dropout entirely
    """
    # Direct linear transformation without dropout
    linear_out = torch.matmul(in_2, in_1.transpose(-2, -1)) + in_0
    
    # Transpose dim 1 and 2: [batch, seq_len, out_features] -> [batch, out_features, seq_len]
    transpose_out = linear_out.transpose(1, 2)
    
    # Return both results to match original pattern
    return (linear_out, transpose_out)

def replacement_func():
    """Return the optimized function"""
    return eliminate_zero_dropout