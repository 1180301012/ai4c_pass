import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    """
    Match the pattern:
    tmp_2 = torch.nn.functional.layer_norm(in_3, (256,), in_1, in_0, 1e-05)
    tmp_4 = tmp_2.sigmoid()
    return tmp_4
    """
    tmp_2 = torch.nn.functional.layer_norm(in_3, (256,), in_1, in_0, 1e-05)
    tmp_4 = tmp_2.sigmoid()
    return tmp_4


def replacement_args(in_0, in_1, in_3):
    # Return (bias, weight, input_layernorm)
    return (in_0, in_1, in_3)


@triton.jit
def layer_norm_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows, normalized_shape, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes elements in a row-parallel manner
    # We use multiple programs to handle different rows
    row_idx = tl.program_id(0)
    
    # Bounds check
    if row_idx >= n_rows:
        return
    
    row_offset = row_idx * normalized_shape
    
    # Load weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Load input for this row
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * normalized_shape
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean using Welfold's online algorithm for numerical stability
    # But for simplicity, use standard approach
    mean = tl.sum(x, axis=0) / normalized_shape
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / normalized_shape
    std = tl.sqrt(var + eps)
    x_norm = x_centered / std
    
    # Apply affine transform and sigmoid
    y = x_norm * weight + bias
    sigmoid_output = 1.0 / (1.0 + tl.exp(-y))
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid_output, mask=mask)


@torch.fx.wrap
def fused_layer_norm_sigmoid(bias, weight, input_layernorm):
    """
    Fused LayerNorm + Sigmoid kernel for better performance.
    input_layernorm: [300, 1, 256]
    weight: [256]
    bias: [256]
    """
    # Reshape input_layernorm to 2D [300, 256]
    input_layernorm_2d = input_layernorm.squeeze(1)  # [300, 256]
    
    n_rows = input_layernorm_2d.shape[0]
    normalized_shape = input_layernorm_2d.shape[1]
    eps = 1e-05
    
    # Fixed block size matching the normalized shape
    BLOCK_SIZE = 256
    
    # Allocate output tensor
    output = torch.empty_like(input_layernorm_2d)
    
    # Launch kernel - one program per row
    layer_norm_sigmoid_kernel[(n_rows,)](
        input_ptr=input_layernorm_2d,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_rows=n_rows,
        normalized_shape=normalized_shape,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output back to original shape
    output = output.unsqueeze(1)
    
    return output


def replacement_func():
    return fused_layer_norm_sigmoid