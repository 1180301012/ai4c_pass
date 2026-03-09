import torch


def pattern(in_0, in_1):
    """Match the rotary embedding computation pattern for arange(64).
    
    This pass targets graph 0 with seq_len=64.
    """
    tmp_0 = in_0
    tmp_1 = torch.arange(64, device=torch.device('cuda:0'))
    tmp_2 = tmp_1.type_as(tmp_0)
    tmp_1 = None
    tmp_3 = torch.outer(tmp_2, tmp_0)
    tmp_2 = tmp_0 = None
    tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
    tmp_3 = None
    tmp_5 = tmp_4.to(torch.device('cuda:0'))
    tmp_4 = None
    tmp_6 = tmp_5.cos()
    tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_6 = None
    tmp_8 = tmp_5.sin()
    tmp_5 = None
    tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = None
    tmp_10 = tmp_7[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
    tmp_11 = tmp_9[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
    tmp_12 = in_1 * tmp_10
    tmp_10 = None
    tmp_13 = in_1.chunk(2, dim=-1)
    tmp_14 = tmp_13[0]
    tmp_15 = tmp_13[1]
    tmp_13 = None
    return tmp_7, tmp_9, tmp_11, tmp_12, tmp_14, tmp_15


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def rotary_embedding_wrapper(in_0, in_1):
    """Optimized rotary embedding computation."""
    device = in_1.device
    inv_freq = in_0
    seq_len = in_1.shape[2]
    N = 64
    
    # Compute positions
    positions = torch.arange(N, device=device, dtype=inv_freq.dtype)
    
    # Compute angles: outer product and concatenate in one expression
    angles = torch.cat([torch.ger(positions, inv_freq), torch.ger(positions, inv_freq)], dim=-1)
    
    # Compute cos/sin more efficiently
    cos_mat = angles.cos()
    sin_mat = angles.sin()
    
    # Add dimensions
    cos_full = cos_mat.unsqueeze(0).unsqueeze(0)
    sin_full = sin_mat.unsqueeze(0).unsqueeze(0)
    
    # Slice
    sin_sliced = sin_full[:, :, :N, :]
    
    # Multiply with query
    mul_result = in_1 * cos_full[:, :, :N, :]
    
    # Chunk
    chunk_0, chunk_1 = in_1.chunk(2, dim=-1)
    
    return cos_full, sin_full, sin_sliced, mul_result, chunk_0, chunk_1


def replacement_func():
    return rotary_embedding_wrapper