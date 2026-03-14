import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: element-wise multiply -> batch_norm (inference) -> silu
    """
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def fused_mul_bn_silu_kernel(
    input_ptr,
    scale_ptr,
    out_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    n_channels,
    n_spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: mul + batch_norm + silu
    Each program handles a block of spatial elements for all channels
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_spatial
    
    # For each spatial position, process all channels
    for c in range(n_channels):
        # Calculate linear index for this channel and spatial positions
        indices = c * n_spatial + offsets
        
        # Load inputs
        x = tl.load(input_ptr + indices, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + c)
        
        # Element-wise multiply
        mul_out = x * scale
        
        # Batch norm (inference mode)
        mean = tl.load(running_mean_ptr + c)
        var = tl.load(running_var_ptr + c)
        w = tl.load(weight_ptr + c)
        b = tl.load(bias_ptr + c)
        
        normalized = (mul_out - mean) / tl.sqrt(var + eps)
        bn_out = normalized * w + b
        
        # SiLU: x * sigmoid(x)
        sigmoid_out = tl.sigmoid(bn_out)
        silu_out = bn_out * sigmoid_out
        
        # Store output
        tl.store(out_ptr + indices, silu_out, mask=mask)


@torch.fx.wrap
def fused_mul_bn_silu(running_mean, running_var, bias, weight, scale, input_tensor):
    """
    Wrapper for the fused mul + batch_norm + silu kernel
    Args:
        running_mean: [C] - batch norm running mean
        running_var: [C] - batch norm running variance
        bias: [C] - batch norm bias
        weight: [C] - batch norm weight
        scale: [B, C, H, W] or [C, H, W] - scale factor (sigmoid output)
        input_tensor: [B, C, H, W] - input tensor
    """
    # Get shape info
    shape = input_tensor.shape
    if len(shape) == 4:
        B, C, H, W = shape
    else:
        raise ValueError(f"Expected 4D input, got shape {shape}")
    
    n_channels = C
    n_spatial = H * W
    eps = 1e-05
    
    # Allocate output
    out = torch.empty_like(input_tensor)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = ((B * n_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Reshape to [B*C, H*W] for easier indexing
    input_flat = input_tensor.reshape(B, C, -1)
    scale_flat = scale.reshape(B, C, -1)
    out_flat = out.reshape(B, C, -1)
    
    # Process each batch element
    for b in range(B):
        fused_mul_bn_silu_kernel[grid](
            input_flat[b],
            scale_flat[b],
            out_flat[b],
            running_mean,
            running_var,
            weight,
            bias,
            n_channels,
            n_spatial,
            eps,
            BLOCK_SIZE,
        )
    
    return out


def replacement_func():
    return fused_mul_bn_silu