import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches adaptive_avg_pool2d applied to intermediate tensor
# Note: Pattern must match exactly without cleanup statements
def pattern(tmp_1):
    return torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)

# Argument extraction function
# Extracts input tensor and spatial dimensions
# Returns (input_tensor, H, W)
def replacement_args(tmp_1):
    H = tmp_1.shape[2]
    W = tmp_1.shape[3]
    return (tmp_1, H, W)

# Optimized Triton kernel for adaptive_avg_pool2d (1x1 case)
@triton.jit
def adaptive_avg_pool_kernel(input_ptr, output_ptr, B, C, H, W, BLOCK_SIZE: tl.constexpr=49):
    block_id = tl.program_id(0)
    batch_idx = block_id // C
    channel_idx = block_id % C
    start_idx = batch_idx * C * H * W + channel_idx * H * W
    sum_val = tl.zeros((1,), dtype=tl.float32)
    thread_id = tl.thread_id(0)
    # Process each spatial element
    if thread_id < H * W:
        idx = start_idx + thread_id
        val = tl.load(input_ptr + idx)
        sum_val = val
    # Block-wide reduction
    sum_val = tl.sum(sum_val, axis=0)
    # Write result to output (only thread 0 does this)
    if thread_id == 0:
        out_idx = batch_idx * C + channel_idx
        tl.store(output_ptr + out_idx, sum_val / (H * W))

# Kernel wrapper with torch.fx.wrap
@torch.fx.wrap
def adaptive_avg_pool2d_kernel_wrapper(input_tensor, H, W):
    B = input_tensor.shape[0]
    C = input_tensor.shape[1]
    grid_size = B * C
    output_tensor = torch.empty((B, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    adaptive_avg_pool_kernel[(grid_size,)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=49
    )
    return output_tensor

# Replacement function (returns the kernel wrapper)
def replacement_func():
    return adaptive_avg_pool2d_kernel_wrapper