import torch
import triton
import triton.language as tl


@triton.jit
def unfold_transpose_reshape_kernel_384(
    in_ptr,
    out_ptr,
    in_batch: tl.constexpr,
    in_length: tl.constexpr,
    orig_channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for channels=384 case
    """
    pid = tl.program_id(0)
    
    # Output shape: [B * L * 2, C/2, 9] where C/2 = 192
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


def unfold_transpose_reshape_384(x):
    """Fused implementation for channels=384 case"""
    B, C, L = x.shape
    
    out_channels = C // 2
    B_out = B * L * 2
    out = torch.empty((B_out, out_channels, 9), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    num_programs = (B_out * out_channels * 9 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    unfold_transpose_reshape_kernel_384[(num_programs,)](
        in_ptr=x,
        out_ptr=out,
        in_batch=B,
        in_length=L,
        orig_channels=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@torch.fx.wrap
def unfold_transpose_reshape_wrapper_384(x):
    return unfold_transpose_reshape_384(x)


def pattern(in_0):
    """
    Match the pattern for channels=384 case
    """
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return unfold_transpose_reshape_wrapper_384