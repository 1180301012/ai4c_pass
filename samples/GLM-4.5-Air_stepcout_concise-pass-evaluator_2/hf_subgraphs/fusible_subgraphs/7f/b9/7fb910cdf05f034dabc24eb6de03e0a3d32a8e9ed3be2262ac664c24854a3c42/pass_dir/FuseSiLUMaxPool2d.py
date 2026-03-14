import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must match the exact computation in model.py
def pattern(in_0):
    # Match the exact sequence: SiLU -> MaxPool2d with same parameters
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return (tmp_1, tmp_0)

# Extract arguments for the replacement
def replacement_args(in_0):
    return (in_0,)

# Triton kernels for fused SiLU + MaxPool2d
@triton.jit
def silu_kernel_forward(x_ptr, silu_out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Element-wise SiLU activation: x * sigmoid(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu_result = x / (1.0 + tl.exp(-x))
    
    # Store SiLU result
    tl.store(silu_out_ptr + offsets, silu_result, mask=mask)

@triton.jit
def max_pool2d_5x1_kernel_2d(
    input_ptr, output_ptr, 
    batch_size, num_channels, height, width,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    """5x5 max pooling with stride=1, padding=2"""
    # Program identifiers
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel
    pid_h = tl.program_id(2)  # block in height
    pid_w = tl.program_id(3)  # block in width
    
    # Compute base pointers
    input_base = input_ptr + (
        pid_b * num_channels * height * width +
        pid_c * height * width
    )
    output_base = output_ptr + (
        pid_b * num_channels * height * width +
        pid_c * height * width
    )
    
    # Load input tile (5x5 neighborhood around current position)
    h_base = pid_h * BLOCK_H
    w_base = pid_w * BLOCK_W
    
    # Load 5x5 neighborhood for all threads in the block
    max_vals = tl.full((BLOCK_H, BLOCK_W), -float('inf'), dtype=tl.float32)
    
    for kh in range(5):
        for kw in range(5):
            # Calculate absolute positions with padding
            abs_h = h_base + tl.arange(0, BLOCK_H) - 2 + kh
            abs_w = w_base + tl.arange(0, BLOCK_W) - 2 + kw
            
            # Create mask for valid positions
            h_mask = (abs_h >= 0) & (abs_h < height)
            w_mask = (abs_w >= 0) & (abs_w < width)
            mask = h_mask[:, None] & w_mask[None, :]
            
            # Load current positions
            ptr_offset = (abs_h[:, None] * width + abs_w[None, :]).to(tl.int64)
            vals = tl.load(input_base + ptr_offset, mask=mask, other=-float('inf'))
            
            # Update max values
            max_vals = tl.maximum(max_vals, vals)
    
    # Store max values
    out_h_start = pid_h * BLOCK_H
    out_w_start = pid_w * BLOCK_W
    out_offsets = (out_h_start[:, None] * width + out_w_start[None, :]).to(tl.int64)
    tl.store(output_base + out_offsets, max_vals, mask=True)

# Fused operation wrapper
@torch.fx.wrap
def fused_silu_maxpool(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    
    # Create output tensors
    silu_result = torch.empty_like(input_tensor, dtype=torch.float32)
    maxpool_result = torch.empty_like(input_tensor, dtype=torch.float32)
    
    n_elements = input_tensor.numel()
    
    # Element-wise SiLU with auto-tuning
    BLOCK_SIZE_SILU = 1024
    num_programs_silu = (n_elements + BLOCK_SIZE_SILU - 1) // BLOCK_SIZE_SILU
    
    silu_kernel_forward[(num_programs_silu,)](
        input_tensor,
        silu_result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE_SILU
    )
    
    # 2D Max Pooling with auto-tuning for different block sizes
    # Choose block size based on spatial dimensions for better occupancy
    if height >= 32 and width >= 32:
        BLOCK_H, BLOCK_W = 16, 16
    elif height >= 16 and width >= 16:
        BLOCK_H, BLOCK_W = 8, 8
    elif height >= 8 and width >= 8:
        BLOCK_H, BLOCK_W = 4, 4
    else:
        BLOCK_H, BLOCK_W = 2, 2
    
    num_blocks_h = (height + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (width + BLOCK_W - 1) // BLOCK_W
    
    max_pool2d_5x1_kernel_2d[(
        batch_size,
        channels,
        num_blocks_h,
        num_blocks_w
    )](
        silu_result,
        maxpool_result,
        batch_size,
        channels,
        height,
        width,
        BLOCK_H,
        BLOCK_W
    )
    
    return maxpool_result, silu_result

# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_silu_maxpool