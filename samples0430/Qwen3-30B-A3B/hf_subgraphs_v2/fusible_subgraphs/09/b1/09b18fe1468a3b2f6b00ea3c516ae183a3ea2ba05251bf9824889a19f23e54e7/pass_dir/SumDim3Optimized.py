import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    return tmp_2.sum(dim = 3)

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def sum_kernel(
    x_ptr,
    y_ptr,
    n_batch,
    n_seq,
    n_feature,
    n_channel,
    BLOCK_SIZE: tl.constexpr
):
    # Each block processes one element along dim=3
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    feature_id = tl.program_id(2)
    
    # Calculate starting offset for this element
    row_start = batch_id * (n_seq * n_feature * n_channel) + \
                seq_id * (n_feature * n_channel) + \
                feature_id * n_channel

    # Load the values to sum (along dim=3)
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < n_batch * n_seq * n_feature * n_channel)

    # Sum along dim=3 (last dimension)
    sum_val = tl.sum(x)
    
    # Store the result
    y_idx = batch_id * (n_seq * n_feature) + seq_id * n_feature + feature_id
    tl.store(y_ptr + y_idx, sum_val)

@torch.fx.wrap
def sum_wrapper(x):
    batch, seq, feature, channel = x.shape
    n_batch = batch
    n_seq = seq
    n_feature = feature
    n_channel = channel
    BLOCK_SIZE = 512  # Reasonable block size
    grid = (n_batch, n_seq, n_feature)
    
    y = torch.empty((n_batch, n_seq, n_feature), dtype=x.dtype, device=x.device)
    sum_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        n_batch=n_batch,
        n_seq=n_seq,
        n_feature=n_feature,
        n_channel=n_channel,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def replacement_func():
    return sum_wrapper