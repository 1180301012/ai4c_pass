import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_1):
    mul = in_1 * in_0
    sum_val = torch.sum(mul, dim=1)
    unsqueezed = sum_val.unsqueeze(1)
    sigmoid_val = torch.sigmoid(unsqueezed)
    return sigmoid_val

# Argument extraction function

def replacement_args(in_0, in_1, mul, sum_val, unsqueezed, sigmoid_val):
    return (mul,)

# Optimized Triton kernel

@triton.jit
def sum_reduce_sigmoid_kernel(
    input_ptr,
    output_ptr,
    N: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_SIZE: tl.constexpr = 64
):
    
    # Calculate spatial position (n, h, w)
    spatial_id = tl.program_id(0)
    if spatial_id >= N * H * W:
        return
    n = spatial_id // (H * W)
    h = (spatial_id % (H * W)) // W
    w = spatial_id % W

    # Only thread 0 computes the full reduction (efficient for small BLOCK_SIZE)
    if tl.thread_id(0) == 0:
        total = 0.0
        # Sum over channel dimension (C)
        for c in range(BLOCK_SIZE):
            # Calculate index for (n, c, h, w)
            input_idx = n * BLOCK_SIZE * H * W + c * H * W + h * W + w
            val = tl.load(input_ptr + input_idx)
            total += val
        
        # Apply sigmoid: 1 / (1 + exp(-total))
        total = 1.0 / (1.0 + tl.exp(-total))
        
        # Store result at (n, 0, h, w) in output
        out_idx = n * H * W + h * W + w
        tl.store(output_ptr + out_idx, total)

# Kernel wrapper
@torch.fx.wrap

def sum_reduce_sigmoid_wrapper(mul):
    # Get tensor dimensions
    N, C, H, W = mul.shape
    
    # Create output tensor with shape [N, 1, H, W]
    output = torch.empty((N, 1, H, W), dtype=mul.dtype, device=mul.device)

    # Configure kernel grid/block
    num_spatial = N * H * W
    num_blocks = (num_spatial + 63) // 64  # Using 64 threads per block

    # Launch kernel
    sum_reduce_sigmoid_kernel[(num_blocks,)](
        mul,
        output,
        N,
        H,
        W,
        BLOCK_SIZE=64
    )

    return output

# Replacement function

def replacement_func():
    return sum_reduce_sigmoid_wrapper