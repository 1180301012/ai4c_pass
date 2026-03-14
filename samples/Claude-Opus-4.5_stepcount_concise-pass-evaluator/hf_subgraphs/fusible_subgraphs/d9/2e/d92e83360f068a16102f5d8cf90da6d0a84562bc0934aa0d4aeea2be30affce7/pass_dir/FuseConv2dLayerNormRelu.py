import torch
import triton
import triton.language as tl


def pattern(conv_bias, conv_weight, ln_bias, ln_weight, x):
    """
    Pattern to match: conv2d -> layer_norm -> relu
    All operations use 1x1 spatial dimensions.
    """
    conv_out = torch.conv2d(x, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    # Note: normalized_shape will be matched symbolically
    ln_out = torch.nn.functional.layer_norm(conv_out, conv_out.shape[1:], ln_weight, ln_bias, 1e-05)
    out = torch.nn.functional.relu(ln_out, inplace=True)
    return (out,)


def replacement_args(conv_bias, conv_weight, ln_bias, ln_weight, x):
    return (conv_bias, conv_weight, ln_bias, ln_weight, x)


@triton.jit
def fused_layernorm_relu_kernel(
    x_ptr,           # Input tensor [N, C, 1, 1]
    ln_weight_ptr,   # LayerNorm weight [C, 1, 1]
    ln_bias_ptr,     # LayerNorm bias [C, 1, 1]
    out_ptr,         # Output tensor [N, C, 1, 1]
    N: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm + ReLU kernel.
    Each program handles one sample (row) in the batch.
    """
    # Program ID corresponds to batch sample index
    row_idx = tl.program_id(0)
    
    # Pointers for this row
    row_start = row_idx * C
    
    # Create offsets for channels
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < C
    
    # Load input row
    x_ptrs = x_ptr + row_start + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / C
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    
    # Normalize
    x_norm = x_centered / tl.sqrt(var + eps)
    
    # Load LayerNorm parameters
    ln_weight = tl.load(ln_weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    ln_bias = tl.load(ln_bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Apply affine transformation
    y = x_norm * ln_weight + ln_bias
    
    # Apply ReLU
    y = tl.maximum(y, 0.0)
    
    # Store output
    out_ptrs = out_ptr + row_start + col_offsets
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_conv_layernorm_relu(conv_bias, conv_weight, ln_bias, ln_weight, x):
    """
    Fused Conv2d + LayerNorm + ReLU implementation.
    Uses PyTorch's conv2d (cuDNN) and custom Triton kernel for LayerNorm + ReLU.
    """
    # Perform conv2d using PyTorch (highly optimized cuDNN)
    conv_out = torch.conv2d(x, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Get dimensions
    N, C, H, W = conv_out.shape
    assert H == 1 and W == 1, "Expected spatial dimensions to be 1x1"
    
    # Flatten spatial dimensions for Triton kernel
    conv_flat = conv_out.view(N, C)
    
    # Flatten LayerNorm parameters
    ln_weight_flat = ln_weight.view(C)
    ln_bias_flat = ln_bias.view(C)
    
    # Allocate output
    out_flat = torch.empty_like(conv_flat)
    
    # Determine block size (next power of 2 >= C)
    BLOCK_SIZE = triton.next_power_of_2(C)
    
    # Launch kernel
    grid = (N,)
    fused_layernorm_relu_kernel[grid](
        conv_flat,
        ln_weight_flat,
        ln_bias_flat,
        out_flat,
        N=N,
        C=C,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to 4D
    out = out_flat.view(N, C, 1, 1)
    return (out,)


def replacement_func():
    return fused_conv_layernorm_relu