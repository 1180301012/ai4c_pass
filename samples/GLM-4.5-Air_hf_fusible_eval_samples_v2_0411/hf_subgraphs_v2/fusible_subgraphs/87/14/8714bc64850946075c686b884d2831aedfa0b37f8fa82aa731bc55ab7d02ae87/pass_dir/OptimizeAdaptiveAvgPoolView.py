import torch
import triton
import triton.language as tl

# Pattern matching function for AdaptiveAvgPool2D optimization
def pattern(tmp_9):
    # AdaptiveAvgPool2D operation with output size (1,1)
    tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
    return tmp_10

# Argument extraction function
def replacement_args(tmp_9):
    return (tmp_9,)

# Triton kernel for optimized global average pooling
@triton.jit
def optimized_global_avg_pool_kernel(
    x_ptr,                  # Input tensor [N, C, H, W]
    output_ptr,             # Output for global pool [N, C, H_out, W_out]
    n_elements,             # Total number of output elements
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    out_H: tl.constexpr,
    out_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Since we're pooling to (1,1), H_out = 1, W_out = 1
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for output [N, C, 1, 1]
    idx = offsets
    n_idx = idx // (C * out_H * out_W)
    rem_idx = idx % (C * out_H * out_W)
    c_idx = rem_idx // (out_H * out_W)
    
    # Calculate sum for this (n, c) pair
    sum_val = 0.0
    total_elements = 0
    
    # Sum over all spatial dimensions
    for h in range(H):
        for w in range(W):
            input_idx = n_idx * C * H * W + c_idx * H * W + h * W + w
            x_val = tl.load(x_ptr + input_idx, mask=mask, other=0.0)
            sum_val += x_val
            total_elements += 1
    
    # Convert to average
    avg_val = sum_val / total_elements
    
    # Store the result at position (n_idx, c_idx, 0, 0)
    output_idx = n_idx * C + c_idx
    tl.store(output_ptr + output_idx, avg_val, mask=mask)

# Kernel wrapper for optimized global average pooling
@torch.fx.wrap
def optimized_global_avg_pool(x):
    N, C, H, W = x.shape
    
    # AdaptiveAvgPool2D to (1,1) outputs [N, C, 1, 1]
    out_shape = (N, C, 1, 1)
    
    # Allocate output tensor
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Total number of elements
    n_elements = N * C * 1 * 1
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_global_avg_pool_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        N=N,
        C=C,
        H=H,
        W=W,
        out_H=1,
        out_W=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_global_avg_pool