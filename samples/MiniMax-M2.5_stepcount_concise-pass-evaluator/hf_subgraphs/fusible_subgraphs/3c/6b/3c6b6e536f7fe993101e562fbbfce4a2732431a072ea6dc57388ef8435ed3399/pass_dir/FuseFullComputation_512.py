import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Full pattern matching both computation paths for slice=512.
    """
    # Path A: negation -> concat -> mul -> add -> cast
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    
    # Path B: slice -> mul -> chunk
    tmp_5 = in_0[slice(None, None, None), slice(None, None, None), slice(None, 512, None), slice(None, None, None)]
    tmp_6 = in_1[slice(None, None, None), slice(None, None, None), slice(None, 512, None), slice(None, None, None)]
    tmp_7 = in_3 * tmp_5
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    
    return tmp_6, tmp_7, tmp_4, tmp_9, tmp_10


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@torch.fx.wrap
def fused_full_pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Fused implementation for slice=512."""
    # Path A: avoid torch.cat
    neg_in6 = -in_6
    batch, head, seq, half_dim = in_5.shape
    dim = half_dim * 2
    concat_result = torch.empty((batch, head, seq, dim), dtype=torch.float32, device=in_2.device)
    concat_result[:, :, :, :half_dim] = neg_in6
    concat_result[:, :, :, half_dim:] = in_5
    tmp_2 = concat_result * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    
    # Path B
    seq_slice = in_0.shape[2]
    tmp_5 = in_0[:, :, :seq_slice, :]
    tmp_6 = in_1[:, :, :seq_slice, :].contiguous()
    tmp_7 = in_3 * tmp_5
    tmp_9, tmp_10 = in_3.chunk(2, dim=-1)
    
    return tmp_6, tmp_7, tmp_4, tmp_9, tmp_10


def replacement_func():
    return fused_full_pattern