import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """Match graph 7 index + expand pattern, return both outputs"""
    # Match the index and expand part
    tmp_4 = in_2[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_5 = tmp_4.expand(64, 4, 4, 128, 128)
    
    # Also match the linear -> view -> transpose part for the second output
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = tmp_1.view((64, 128, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    
    return tmp_5, tmp_3


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1, in_2)


@torch.fx.wrap
def optimized_computation_7(in_0, in_1, in_2):
    """Optimized implementation using matmul instead of linear"""
    # Compute linear via matmul
    in_1_flat = in_1.reshape(-1, in_1.shape[-1])  # (batch*seq, hidden)
    weight = in_0.t()  # (hidden, out_features)
    tmp_1 = torch.matmul(in_1_flat, weight)  # (batch*seq, out_features)
    
    # Reshape back
    batch, seq, _ = in_1.shape
    tmp_1 = tmp_1.view(batch, seq, 512)
    
    # View and transpose
    tmp_2 = tmp_1.view((64, 128, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    
    # Optimize index + expand using unsqueeze
    tmp_4 = in_2.unsqueeze(2)
    tmp_5 = tmp_4.expand(64, 4, 4, 128, 128)
    
    return tmp_5, tmp_3


def replacement_func():
    """Return the replacement function"""
    return optimized_computation_7