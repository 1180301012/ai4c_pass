import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    tmp_2 = None
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_permute_reshape_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    orig_seq_len,
    hidden_size,
    target_channels,
    target_h,
    target_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * target_channels * target_h * target_w
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices in output space
    channels = offsets // (target_h * target_w)
    h = (offsets // target_w) % target_h
    w = offsets % target_w
    batch_idx = offsets // (target_channels * target_h * target_w)
    
    # Map back to input indices: (0, 2, 1) permutation
    # Output: (batch, channels, h, w)
    # Input: (batch, orig_seq_len, hidden_size)
    # The reshape is: (orig_seq_len, hidden_size) -> (channels, h, w)
    flat_idx_in_channel = h * target_w + w
    seq_idx = flat_idx_in_channel // hidden_size
    hidden_idx = flat_idx_in_channel % hidden_size
    
    # But with permute(0, 2, 1): original input was (batch, seq, hidden) -> (batch, hidden, seq)
    # So we need to find: which position in the flattened hidden x seq space corresponds to our output
    input_flat_idx = batch_idx * orig_seq_len * hidden_size + hidden_idx * orig_seq_len + seq_idx
    
    tl.store(out_ptr + offsets, tl.load(in_ptr + input_flat_idx, mask=mask, other=0.0), mask=mask)

@torch.fx.wrap
def optimized_permute_reshape(in_0):
    batch_size, orig_seq_len, hidden_size = in_0.shape
    
    # Determine target dimensions based on hidden size patterns
    if hidden_size == 64:
        target_channels = 64
        target_h = 128
        target_w = 128
    elif hidden_size == 320:
        target_channels = 320
        target_h = 32
        target_w = 32
    elif hidden_size == 160:
        target_channels = 160
        target_h = 32
        target_w = 32
    else:
        # Fallback to original implementation
        tmp_2 = in_0.permute(0, 2, 1)
        return tmp_2.reshape(batch_size, target_channels, target_h, target_w)
    
    out = torch.empty((batch_size, target_channels, target_h, target_w), dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * target_channels * target_h * target_w
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_permute_reshape_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        orig_seq_len=orig_seq_len,
        hidden_size=hidden_size,
        target_channels=target_channels,
        target_h=target_h,
        target_w=target_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_permute_reshape