import torch
import triton
import triton.language as tl

def pattern(conv_out, ln_weight, ln_bias):
    """
    Pattern: Conv2d output -> view -> permute -> layer_norm -> permute -> view -> dropout -> view -> permute
    This fuses the entire chain after conv2d.
    """
    tmp_6 = conv_out.view(1, 384, 576)
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (384,), ln_weight, ln_bias, 1e-05)
    tmp_9 = tmp_8.permute(0, 2, 1)
    tmp_10 = tmp_9.view(1, 384, 24, 24)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
    tmp_12 = tmp_11.view(1, 384, 576)
    tmp_13 = tmp_12.permute(0, 2, 1)
    return tmp_13

def replacement_args(conv_out, ln_weight, ln_bias):
    return (conv_out, ln_weight, ln_bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}),
    ],
    key=['C'],
)
@triton.jit
def fused_transpose_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C,  # Number of channels (384)
    H,  # Height (24)
    W,  # Width (24)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one spatial position (h, w)
    # Input layout: (1, C, H, W) with shape (1, 384, 24, 24)
    # Output layout: (1, H*W, C) with shape (1, 576, 384)
    
    spatial_idx = tl.program_id(0)  # 0 to H*W-1
    
    # Load all channels for this spatial position
    c_offsets = tl.arange(0, BLOCK_SIZE)
    c_mask = c_offsets < C
    
    # Input is in (C, H, W) layout, we need to read all C values for this spatial position
    # spatial_idx = h * W + w
    # For channel c, the offset is: c * H * W + spatial_idx
    input_offsets = c_offsets * H * W + spatial_idx
    x = tl.load(input_ptr + input_offsets, mask=c_mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / C
    
    # Compute variance
    x_centered = tl.where(c_mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / C
    
    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rstd
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1.0)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    output = x_norm * weight + bias
    
    # Store result in (H*W, C) layout
    # For spatial position spatial_idx and channel c, offset is: spatial_idx * C + c
    output_offsets = spatial_idx * C + c_offsets
    tl.store(output_ptr + output_offsets, output, mask=c_mask)

@torch.fx.wrap
def fused_conv_layernorm_reshape(conv_out, ln_weight, ln_bias):
    """
    Fused kernel that:
    1. Reads from conv_out with shape (1, 384, 24, 24)
    2. Applies layer normalization across the channel dimension
    3. Writes to output with shape (1, 576, 384)
    
    This avoids all intermediate PyTorch reshape operations.
    """
    batch, C, H, W = conv_out.shape
    
    # Output shape: (1, H*W, C) = (1, 576, 384)
    output = torch.empty((batch, H * W, C), dtype=conv_out.dtype, device=conv_out.device)
    
    eps = 1e-05
    grid = (H * W,)  # 576 programs
    
    # Remove batch dimension for kernel (we know batch=1)
    conv_out_2d = conv_out.squeeze(0)  # Shape: (384, 24, 24)
    output_2d = output.squeeze(0)  # Shape: (576, 384)
    
    fused_transpose_layernorm_kernel[grid](
        conv_out_2d,
        ln_weight,
        ln_bias,
        output_2d,
        C,
        H,
        W,
        eps,
    )
    
    return output

def replacement_func():
    return fused_conv_layernorm_reshape