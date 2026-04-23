import torch
import triton
import triton.language as tl


@triton.jit
def unfold_transpose_reshape_kernel(
    in_ptr,
    out_ptr,
    in_batch: tl.constexpr,
    in_length: tl.constexpr,
    orig_channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    unsqueeze(-1) + unfold(kernel=[9,1], pad=[4,0], stride=1) + transpose(1,2) + reshape + reshape
    
    Input shape: [batch, channels, length]
    Output shape: [batch * length * 2, channels/2, 9]
    """
    pid = tl.program_id(0)
    
    # Output shape: [B * L * 2, C/2, 9]
    out_channels = orig_channels // 2
    total_out_elements = in_batch * in_length * 2 * out_channels * 9
    
    if pid * BLOCK_SIZE >= total_out_elements:
        return
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out_elements
    
    # Decode flat offset to output coordinates
    k = offsets % 9
    c_out = (offsets // 9) % out_channels
    pair_idx = (offsets // (9 * out_channels)) % 2
    l_out = (offsets // (9 * out_channels * 2)) % in_length
    batch_out = (offsets // (9 * out_channels * 2 * in_length)) % in_batch
    
    # Original channel for this element
    c_orig = pair_idx * out_channels + c_out
    
    # For unfold with kernel=[9,1], pad=[4,0], stride=1
    h_pos = l_out + k - 4  # pad_h = 4
    
    # Check bounds
    valid = (h_pos >= 0) and (h_pos < in_length)
    
    # Calculate source offset
    src_offset = batch_out * orig_channels * in_length + c_orig * in_length + h_pos
    
    # Load value
    val = tl.load(in_ptr + src_offset, mask=mask & valid, other=0.0)
    
    # Store to output
    tl.store(out_ptr + offsets, val, mask=mask)


def identity_op(x):
    """Identity operation for testing pattern matching"""
    return x


@torch.fx.wrap
def unfold_transpose_reshape_wrapper(x):
    return unfold_transpose_reshape(x)


def pattern(in_0):
    tmp_0 = in_0.reshape(1, -1, 16, 9)
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return identity_op