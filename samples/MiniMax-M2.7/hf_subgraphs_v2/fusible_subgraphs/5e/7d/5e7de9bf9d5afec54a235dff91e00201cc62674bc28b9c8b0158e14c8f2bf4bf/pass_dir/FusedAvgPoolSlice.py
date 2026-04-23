import torch
import triton
import triton.language as tl

# ============================================================================
# Fused AvgPool1d + Slice Pass
# ============================================================================
# This pass fuses avg_pool1d and the slice operation for the avg_pool branch.
# Input: in_3 [1, 768, 249]
# Output: tmp_6 [1, 768, 124]

@triton.jit
def fused_avgpool_slice_kernel(
    in_ptr, out_ptr,
    batch_size, channels, in_len,
    kernel_size, stride, padding,
    out_len,  # = 124
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // (channels * out_len)
    ch = (pid // out_len) % channels
    out_pos = pid % out_len
    
    # Input start position for average pooling
    in_start = out_pos * stride - padding
    
    # Accumulate for average pooling
    acc = 0.0
    count = 0
    for k in range(kernel_size):
        in_pos = in_start + k
        # Check bounds: need both conditions
        is_valid = (in_pos >= 0) and (in_pos < in_len)
        if is_valid:
            acc += tl.load(in_ptr + (batch * channels * in_len + ch * in_len + in_pos))
            count += 1
    
    # Average
    out_val = acc / tl.cast(kernel_size, tl.float32)
    
    # Store result
    out_idx = batch * channels * out_len + ch * out_len + out_pos
    tl.store(out_ptr + out_idx, out_val)


@torch.fx.wrap
def fused_avgpool_slice(in_3):
    """Fused avg_pool1d + slice[..., :124] kernel wrapper"""
    batch_size, channels, in_len = in_3.shape  # [1, 768, 249]
    
    # avg_pool1d parameters: kernel_size=2, stride=2, padding=0
    kernel_size, stride, padding = 2, 2, 0
    out_len_full = (in_len + 2 * padding - kernel_size) // stride + 1  # 124
    out_len = 124  # After slicing
    
    total_elements = batch_size * channels * out_len
    
    # Allocate output
    out = torch.empty((batch_size, channels, out_len), 
                      dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    BLOCK_SIZE = 128
    num_programs = total_elements
    
    fused_avgpool_slice_kernel[(num_programs,)](
        in_ptr=in_3, out_ptr=out,
        batch_size=batch_size, channels=channels, in_len=in_len,
        kernel_size=kernel_size, stride=stride, padding=padding,
        out_len=out_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_3):
    """Match: avg_pool1d -> slice[..., :124]"""
    tmp_5 = torch.nn.functional.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
    tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
    return tmp_6


def replacement_args(in_3):
    """Extract arguments for the fused kernel"""
    return (in_3,)


def replacement_func():
    """Return the fused avgpool+slice kernel"""
    return fused_avgpool_slice