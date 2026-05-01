import torch
import triton
import triton.language as tl

# Pattern matching function for mean followed by batch_norm
@torch.fx.wrap
def pattern(tmp_4, in_0, in_1, in_2, in_3, in_5):
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (tmp_8, tmp_5)

# Extract necessary arguments
@torch.fx.wrap
def replacement_args(tmp_4, in_0, in_1, in_2, in_3, in_5):
    return (tmp_4, in_5, in_0, in_1, in_3, in_2)

# Triton kernel for fused mean + batch_norm
@triton.jit
def fused_mean_batch_norm_kernel(
    in_4_ptr,
    in_5_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    # Each program handles one channel (C)
    c = tl.program_id(0)
    c_start = c * BLOCK_C
    c_end = min(c_start + BLOCK_C, C)
    c_ids = tl.arange(0, BLOCK_C) + c_start
    c_mask = c_ids < C

    # Load parameters for this channel range
    running_mean = tl.load(running_mean_ptr + c_ids, mask=c_mask)
    running_var = tl.load(running_var_ptr + c_ids, mask=c_mask)
    weight = tl.load(weight_ptr + c_ids, mask=c_mask)
    bias = tl.load(bias_ptr + c_ids, mask=c_mask)

    # Compute spatial mean for channel range
    sum_in4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_in5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for h in range(H):
        for w in range(W):
            # Load in_4 and in_5 for current (h, w)
            in4 = tl.load(in_4_ptr + c_ids[:, None] * H * W + h * W + w, mask=c_mask[:, None], other=0.0)
            in5 = tl.load(in_5_ptr + c_ids[:, None] * H * W + h * W + w, mask=c_mask[:, None], other=0.0)
            sum_in4 += in4
            sum_in5 += in5
    spatial_mean = (sum_in4 + sum_in5) / (H * W)

    # Apply batch norm transformation
    denom = tl.sqrt(running_var + eps)
    normalized = (spatial_mean - running_mean) / denom
    output = normalized * weight + bias

    # Store results
    tl.store(mean_ptr + c_ids, spatial_mean, mask=c_mask)
    tl.store(out_ptr + c_ids, output, mask=c_mask)


# Kernel wrapper
@torch.fx.wrap
def fused_mean_batch_norm(in_4, in_5, running_mean, running_var, weight, bias):
    B, C, H, W = in_4.shape
    # Allocate output tensors
    mean_out = torch.empty(C, dtype=in_4.dtype, device=in_4.device)
    output = torch.empty(C, dtype=in_4.dtype, device=in_4.device)

    # Launch kernel
    grid = (C,)
    BLOCK_C = 32
    fused_mean_batch_norm_kernel[grid](
        in_4_ptr=in_4,
        in_5_ptr=in_5,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        mean_ptr=mean_out,
        out_ptr=output,
        B=B,
        C=C,
        H=H,
        W=W,
        eps=1e-05,
        BLOCK_C=BLOCK_C,
    )
    return output, mean_out

# Replacement function
@torch.fx.wrap
def replacement_func():
    return fused_mean_batch_norm