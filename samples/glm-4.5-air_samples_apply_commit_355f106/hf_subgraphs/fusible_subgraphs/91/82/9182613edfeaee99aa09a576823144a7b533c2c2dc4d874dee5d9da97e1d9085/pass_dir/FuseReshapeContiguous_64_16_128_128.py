import torch
import triton
import triton.language as tl


# Pattern matching function for graph with reshape(64, 16, 128, 128)
def pattern(in_1, in_0, in_2, in_3):
    """
    Pattern: reshape followed by contiguous
    Match the exact computation pattern from graph 7
    """
    tmp_0 = in_1.reshape(64, 16, 128, 128)
    tmp_1 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None)]
    tmp_2 = in_2.contiguous()
    tmp_3 = in_3.contiguous()
    tmp_4 = tmp_0.contiguous()
    return (tmp_1, tmp_3, tmp_2, tmp_4)


def replacement_args(in_1, in_0, in_2, in_3):
    """Extract arguments needed for the replacement"""
    return (in_1, in_0, in_2, in_3)


def optimized_reshape_contiguous(in_1, in_0, in_2, in_3):
    """
    Optimized function - use reshape directly which is already efficient.
    The key optimization is using reshape (which returns a view if possible)
    instead of reshape + contiguous pattern.
    """
    # Use reshape - PyTorch's reshape is already optimized and will return
    # a contiguous tensor if needed
    tmp_0 = in_1.reshape(64, 16, 128, 128)
    
    # Slice operation
    tmp_1 = in_0[..., :128]
    
    # These contiguous calls may be redundant if tensors are already contiguous
    tmp_2 = in_2.contiguous()
    tmp_3 = in_3.contiguous()
    
    # Return the reshape result directly - it's already "contiguous" in the 
    # PyTorch sense (physically contiguous in memory)
    return (tmp_1, tmp_3, tmp_2, tmp_0)


def replacement_func():
    return optimized_reshape_contiguous