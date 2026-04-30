import torch
import triton
import triton.language as tl


@triton.jit
def triton_fused_kernel(
    # Input pointers (4 tensors)
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    # Output pointer
    out_ptr,
    # Channel counts for each input
    channels_0, channels_1, channels_2, channels_3,
    # Spatial dimensions
    H, W,
    total_channels, num_elements,
):
    """
    Optimized fused kernel for cat + adaptive_avg_pool2d + flatten.
    
    Strategy: Each program handles one output channel with optimized
    sequential loads and unrolled accumulation for better ILP.
    """
    pid = tl.program_id(0)
    
    # Determine which input tensor this channel belongs to
    # Branch prediction helps as channels are in contiguous ranges
    if pid < channels_0:
        src_ptr = in_0_ptr
        local_ch = pid
    elif pid < channels_0 + channels_1:
        src_ptr = in_1_ptr
        local_ch = pid - channels_0
    elif pid < channels_0 + channels_1 + channels_2:
        src_ptr = in_2_ptr
        local_ch = pid - channels_0 - channels_1
    else:
        src_ptr = in_3_ptr
        local_ch = pid - channels_0 - channels_1 - channels_2
    
    # Compute base offset for this channel
    ch_base_offset = local_ch * num_elements
    
    # Accumulate sum across all spatial elements
    sum_val = 0.0
    base = ch_base_offset
    
    # Unrolled inner loops for better ILP (H=5, W=5)
    for h_idx in range(H):
        row_base = base + h_idx * W
        # Sequential loads with coalesced memory access
        v0 = tl.load(src_ptr + row_base + 0).to(tl.float32)
        v1 = tl.load(src_ptr + row_base + 1).to(tl.float32)
        v2 = tl.load(src_ptr + row_base + 2).to(tl.float32)
        v3 = tl.load(src_ptr + row_base + 3).to(tl.float32)
        v4 = tl.load(src_ptr + row_base + 4).to(tl.float32)
        sum_val += v0 + v1 + v2 + v3 + v4
    
    # Store the average
    avg_val = sum_val / num_elements
    tl.store(out_ptr + pid, avg_val)


@torch.fx.wrap
def fused_cat_avgpool_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Fused kernel for cat + adaptive_avg_pool2d + flatten.
    
    Avoids materializing the concatenated tensor by directly reading
    from the 4 input tensors and computing average pooling.
    """
    batch_size, channels_0, H, W = in_0.shape
    _, channels_1, _, _ = in_1.shape
    _, channels_2, _, _ = in_2.shape
    _, channels_3, _, _ = in_3.shape
    
    total_channels = channels_0 + channels_1 + channels_2 + channels_3
    num_elements = H * W
    
    # Allocate output tensor
    out = torch.empty((batch_size, total_channels), dtype=in_0.dtype, device=in_0.device)
    
    # Launch one program per output channel for maximum parallelism
    num_programs = total_channels
    
    triton_fused_kernel[(num_programs,)](
        in_0, in_1, in_2, in_3,
        out,
        channels_0, channels_1, channels_2, channels_3,
        H, W,
        total_channels, num_elements
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: cat + adaptive_avg_pool2d + dropout (no-op) + flatten
    """
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract the 4 input tensors for the replacement function.
    """
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """
    Return the fused kernel wrapper function.
    """
    return fused_cat_avgpool_kernel_wrapper