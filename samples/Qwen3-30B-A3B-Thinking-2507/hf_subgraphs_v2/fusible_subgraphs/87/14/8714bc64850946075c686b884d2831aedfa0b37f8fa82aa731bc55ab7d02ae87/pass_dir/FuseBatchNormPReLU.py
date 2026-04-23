import torch
import triton
import triton.language as tl

# Pattern matching function for BatchNorm followed by PReLU
# Must mirror the exact operation sequence from the model
# Note: Parameters must be in the exact order as the model uses

def pattern(x, in1, in2, in4, in3, in0):
    # Match BatchNorm with specific parameters
    bn_out = torch.nn.functional.batch_norm(x, in1, in2, in4, in3, False, 0.1, 0.001)
    # Match PReLU using the same channel dimension (128 channels)
    prelu_out = torch.prelu(bn_out, in0)
    return prelu_out

# Argument extraction (must return the exact parameters needed for kernel)
def replacement_args(x, in1, in2, in4, in3, in0):
    return (x, in1, in2, in4, in3, in0)

# Triton kernel for fused BatchNorm + PReLU
@triton.jit
def fused_batchnorm_prelu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    slope_ptr,
    out_ptr,
    n_batch,
    H,
    W,
    C,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Process each channel independently
    c = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_batch * H * W)

    # Load channel-specific parameters
    running_mean = tl.load(running_mean_ptr + c)
    running_var = tl.load(running_var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    slope = tl.load(slope_ptr + c)

    # Compute BatchNorm normalization
    denominator = tl.sqrt(tl.cast(running_var, tl.float32) + tl.cast(eps, tl.float32))
    bn_scale = weight / denominator
    bn_shift = (bias - running_mean * weight) / denominator

    # Calculate offset to channel c in the tensor
    x_offset = c * H * W * n_batch
    x_ptr += x_offset
    out_ptr += x_offset

    # Load input values for the current channel
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply BatchNorm + PReLU in single pass
    bn_vals = bn_scale * x_vals + bn_shift
    out_vals = tl.where(bn_vals > 0, bn_vals, slope * bn_vals)

    # Store result
    tl.store(out_ptr + offsets, out_vals, mask=mask)


# Kernel wrapper - must be decorated as @torch.fx.wrap
@torch.fx.wrap
def fused_batchnorm_prelu(x, running_mean, running_var, weight, bias, slope):
    # Ensure inputs are contiguous for kernel access
    x = x.contiguous()
    n_batch, channels, H, W = x.shape
    eps = 0.001
    BLOCK_SIZE = 512  # Optimized for 128-channel tensors
    num_blocks = (n_batch * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (channels, num_blocks)

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch kernel
    fused_batchnorm_prelu_kernel[grid](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        slope_ptr=slope,
        out_ptr=out,
        n_batch=n_batch,
        H=H,
        W=W,
        C=channels,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# Replacement function

def replacement_func():
    return fused_batchnorm_prelu