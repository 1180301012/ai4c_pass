import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_6, in_0, in_1, in_2, in_3):
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return (tmp_8,)

# Argument extraction function
def replacement_args(tmp_6, in_0, in_1, in_2, in_3):
    return (tmp_6, in_0, in_1, in_2, in_3)

# Optimized kernel
@triton.jit
def batchnorm_relu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Each block handles one channel
    c = tl.program_id(0)
    # Grid dimensions: H and W
    block_start_h = tl.program_id(1) * BLOCK_H
    block_start_w = tl.program_id(2) * BLOCK_W
    # Thread indices for spatial
    h = block_start_h + tl.arange(0, BLOCK_H)
    w = block_start_w + tl.arange(0, BLOCK_W)
    mask_h = h < H
    mask_w = w < W
    # Get the data for this channel
    rm = tl.load(running_mean_ptr + c)
    rv = tl.load(running_var_ptr + c)
    w_val = tl.load(weight_ptr + c)
    b_val = tl.load(bias_ptr + c)
    # Iterate over the batch size
    for batch in range(N):
        # Calculate the offset for this batch and channel
        x_batch_start = batch * C * H * W + c * H * W
        out_batch_start = batch * C * H * W + c * H * W
        # Load the input for this batch and channel
        x = tl.load(x_ptr + x_batch_start + (h[:, None] * W + w[None, :]), mask=mask_h[:, None] & mask_w[None, :], other=0.0)
        # Compute batchnorm and relu
        normalized = (x - rm) / tl.sqrt(rv + eps)
        out_val = normalized * w_val + b_val
        relu_val = tl.maximum(out_val, 0.0)
        # Store the result
        tl.store(out_ptr + out_batch_start + (h[:, None] * W + w[None, :]), relu_val, mask=mask_h[:, None] & mask_w[None, :])

# Kernel wrapper
@torch.fx.wrap
def batchnorm_relu(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    eps = 0.001
    BLOCK_H = 16
    BLOCK_W = 16
    grid = (C, (H + BLOCK_H - 1) // BLOCK_H, (W + BLOCK_W - 1) // BLOCK_W)
    out = torch.empty_like(x)
    batchnorm_relu_kernel[grid](
        x, running_mean, running_var, weight, bias, out, 
        N, C, H, W, eps, BLOCK_H, BLOCK_W
    )
    return out

# Replacement function
def replacement_func():
    return batchnorm_relu