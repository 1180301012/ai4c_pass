import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE_N': 256}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE_N': 512}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 2048}, num_stages=1, num_warps=4),
    ],
    key=['out_features'],
)
@triton.jit
def fused_reshape_layernorm_kernel(
    # Input pointer (after conv, already transposed: B, N, C)
    input_ptr,
    # Output pointer
    output_ptr,
    # Shapes
    batch_size, seq_len, out_features,
    # Strides
    input_stride_b, input_stride_n, input_stride_c,
    output_stride_b, output_stride_n, output_stride_c,
    # Layer norm params
    norm_weight_ptr, norm_bias_ptr,
    eps,
    # Meta
    BLOCK_SIZE_N: tl.constexpr,
    out_features: tl.constexpr,
):
    # Each program processes one element in the sequence dimension
    # Grid: (batch_size, seq_len)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Compute offsets
    offs_c = tl.arange(0, BLOCK_SIZE_N)
    mask_c = offs_c < out_features
    
    # Load input: (batch, seq, channel)
    input_base = batch_idx * input_stride_b + seq_idx * input_stride_n
    input_vals = tl.load(input_ptr + input_base + offs_c, mask=mask_c, other=0.0)
    
    # Compute mean
    sum_vals = tl.sum(input_vals, axis=0)
    mean = sum_vals / out_features
    
    # Compute variance: E[x^2] - E[x]^2
    sum_sq = tl.sum(input_vals * input_vals, axis=0)
    var = sum_sq / out_features - mean * mean
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (input_vals - mean) * inv_std
    
    # Apply layer norm weight and bias
    if norm_weight_ptr is not None:
        norm_weight = tl.load(norm_weight_ptr + offs_c, mask=mask_c, other=0.0)
        normalized = normalized * norm_weight
    
    if norm_bias_ptr is not None:
        norm_bias = tl.load(norm_bias_ptr + offs_c, mask=mask_c, other=0.0)
        normalized = normalized + norm_bias
    
    # Store output
    output_base = batch_idx * output_stride_b + seq_idx * output_stride_n
    tl.store(output_ptr + output_base + offs_c, normalized, mask=mask_c)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match conv2d + flatten(2) + transpose(1,2) + layer_norm pattern"""
    tmp_0 = in_0  # norm bias
    tmp_1 = in_1  # norm weight
    tmp_2 = in_2  # conv bias
    tmp_3 = in_3  # conv weight
    tmp_4 = in_4  # cls token
    tmp_5 = in_5  # input
    
    # Get kernel size from weight shape (last two dimensions)
    kernel_size = (tmp_3.shape[2], tmp_3.shape[3])
    
    # Conv2D - match both (2,2) and (4,4) kernel sizes
    tmp_6 = torch.conv2d(tmp_5, tmp_3, tmp_2, kernel_size, (0, 0), (1, 1), 1)
    tmp_5 = tmp_3 = tmp_2 = None
    
    # Flatten
    tmp_7 = tmp_6.flatten(2)
    tmp_6 = None
    
    # Transpose
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_7 = None
    
    # LayerNorm - get normalized_shape from the weight
    normalized_shape = tmp_1.shape[0]
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (normalized_shape,), tmp_1, tmp_0, 1e-05)
    tmp_8 = tmp_1 = tmp_0 = None
    
    # Expand
    tmp_10 = tmp_4.expand(1, -1, -1)
    tmp_4 = None
    
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@torch.fx.wrap
def fused_kernel_wrapper(norm_bias, norm_weight, conv_bias, conv_weight, cls_token, input_tensor):
    """Fused conv2d + flatten + transpose + layer_norm kernel
    
    Uses PyTorch's optimized conv2d (cuDNN) and fuses the reshape + layer_norm in Triton.
    """
    
    # Get shapes
    batch_size = input_tensor.shape[0]
    in_channels = input_tensor.shape[1]
    height = input_tensor.shape[2]
    width = input_tensor.shape[3]
    
    out_channels = conv_weight.shape[0]
    kernel_h = conv_weight.shape[2]
    kernel_w = conv_weight.shape[3]
    
    # Compute output height and width after convolution
    # Assuming stride=1, padding=0, dilation=1
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1
    seq_len = out_h * out_w
    
    # Use PyTorch's highly optimized conv2d (cuDNN)
    # This is faster than implementing our own in Triton
    conv_out = torch.conv2d(input_tensor, conv_weight, conv_bias, 
                           (kernel_h, kernel_w), (0, 0), (1, 1), 1)
    
    # Reshape: (B, C, H, W) -> (B, H*W, C) = (B, seq_len, C)
    # First flatten(2): (B, C, H*W)
    conv_flat = conv_out.flatten(2)
    # Then transpose: (B, H*W, C)
    conv_transposed = conv_flat.transpose(1, 2)
    
    # Now apply layer norm using Triton kernel
    # Output shape: (batch, seq_len, out_channels)
    output = torch.empty((batch_size, seq_len, out_channels), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # For small channels, use smaller block sizes
    if out_channels <= 128:
        block_size = 128
    elif out_channels <= 256:
        block_size = 256
    elif out_channels <= 512:
        block_size = 512
    elif out_channels <= 1024:
        block_size = 1024
    else:
        block_size = 2048
    
    # Grid: (batch_size, seq_len)
    grid = (batch_size, seq_len)
    
    fused_reshape_layernorm_kernel[grid](
        conv_transposed,
        output,
        batch_size, seq_len, out_channels,
        conv_transposed.stride(0), conv_transposed.stride(1), conv_transposed.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        norm_weight, norm_bias,
        1e-05,
        BLOCK_SIZE_N=block_size,
        out_features=out_channels,
    )
    
    # Expand cls_token: (1, 1, hidden_dim) -> (1, 1, hidden_dim)
    expanded_cls = cls_token.expand(1, -1, -1)
    
    return (expanded_cls, output)


def replacement_func():
    return fused_kernel_wrapper