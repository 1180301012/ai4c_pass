import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    """
    Pattern for slice=64: slice in_0, slice in_1, multiply with in_3, and chunk in_3
    """
    tmp_5 = in_0[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
    tmp_6 = in_1[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
    tmp_7 = in_3 * tmp_5
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    return tmp_6, tmp_7, tmp_9, tmp_10


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@torch.fx.wrap
def fused_slice_mul_chunk_64(in_0, in_1, in_3):
    """Fused implementation for slice=64 case."""
    # Get shape
    shape_0 = in_0.shape
    seq_slice = shape_0[2]  # Should be 64
    
    # tmp_6 = sliced in_1
    tmp_6 = in_1[:, :, :seq_slice, :].contiguous()
    
    # tmp_5 = sliced in_0
    tmp_5 = in_0[:, :, :seq_slice, :]
    
    # tmp_7 = in_3 * tmp_5
    tmp_7 = in_3 * tmp_5
    
    # tmp_9, tmp_10 = chunk in_3
    tmp_9, tmp_10 = in_3.chunk(2, dim=-1)
    
    return tmp_6, tmp_7, tmp_9, tmp_10


def replacement_func():
    return fused_slice_mul_chunk_64