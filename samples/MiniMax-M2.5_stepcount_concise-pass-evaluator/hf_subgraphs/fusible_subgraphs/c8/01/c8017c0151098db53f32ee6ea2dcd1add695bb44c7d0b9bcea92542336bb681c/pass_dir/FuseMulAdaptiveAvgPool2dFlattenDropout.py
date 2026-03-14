import torch
import triton
import triton.language as tl


# Pattern matching: multiply + adaptive_avg_pool2d + flatten + dropout(p=0)
# Since dropout with p=0 is a no-op, we can fuse these operations
def pattern(in_2, tmp_3):
    """
    Pattern: multiply(in_2, tmp_3) -> adaptive_avg_pool2d -> flatten -> dropout(p=0)
    
    Only return the final output (dropout result) since intermediate values
    are not observable outside this pattern.
    """
    # Multiply in_2 (B, C, H, W) with tmp_3 (B, C, 1, 1) to get tmp_4 (B, C, H, W)
    tmp_4 = in_2 * tmp_3
    
    # Adaptive average pooling to (1, 1)
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    
    # Flatten from dim 1
    tmp_6 = tmp_5.flatten(1, -1)
    
    # Dropout with p=0 is a no-op
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    
    # Only return tmp_7 (the final output) since intermediate values are not
    # observable outside this pattern
    return tmp_7


def replacement_args(in_2, tmp_3):
    return (in_2, tmp_3)


# Autotune configurations with powers of 2 only
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 16}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=4),
    ],
    key=['H', 'W'],
)
@triton.jit
def fused_mul_pool_flatten_kernel(
    in_2_ptr,  # Main input (B, C, H, W)
    tmp_3_ptr,  # Scale factor (B, C, 1, 1)
    output_ptr,  # Output (B, C)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Multiply in_2 * tmp_3 (with broadcasting)
    2. Adaptive average pooling to (1, 1)
    3. Flatten to (B, C)
    """
    # Get batch and channel from program_id
    program_id = tl.program_id(0)
    pid_b = program_id // C
    pid_c = program_id % C
    
    # Calculate base offset for this batch and channel
    base_offset = pid_b * C * H * W + pid_c * H * W
    
    # Load the scale factor
    scale = tl.load(tmp_3_ptr + pid_b * C + pid_c)
    
    # Compute sum over H*W elements
    num_elements = H * W
    
    # Create offsets for this program
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load all elements and multiply by scale
    load_offsets = base_offset + offsets
    vals = tl.load(in_2_ptr + load_offsets, mask=mask, other=0.0)
    
    # Sum the values * scale
    sum_val = tl.sum(vals * scale, axis=0)
    
    # Compute average
    avg = sum_val / tl.constexpr(H * W)
    
    # Store output
    out_offset = pid_b * C + pid_c
    tl.store(output_ptr + out_offset, avg)


@torch.fx.wrap
def fused_mul_pool_flatten_wrapper(in_2, tmp_3):
    """
    Wrapper function that launches the fused kernel.
    """
    B, C, H, W = in_2.shape
    output = torch.empty((B, C), dtype=torch.float32, device=in_2.device)
    
    # 1D grid: B * C programs
    grid = (B * C,)
    
    # Ensure contiguous storage
    in_2 = in_2.contiguous()
    tmp_3 = tmp_3.contiguous()
    
    # Launch kernel
    fused_mul_pool_flatten_kernel[grid](
        in_2_ptr=in_2,
        tmp_3_ptr=tmp_3,
        output_ptr=output,
        B=B,
        C=C,
        H=H,
        W=W,
    )
    
    return output


def replacement_func():
    return fused_mul_pool_flatten_wrapper