import torch
import triton
import triton.language as tl

# ============================================================================
# Fused Conv2D + Add + Flatten + Transpose Triton Kernel
# ============================================================================
# This kernel fuses the following operations:
# 1. depthwise Conv2D with groups=C (produces [B, C, H, W])
# 2. Element-wise addition with input (residual connection)
# 3. Flatten spatial dimensions (produces [B, C, H*W])
# 4. Transpose to [B, H*W, C] for subsequent LayerNorm
#
# The fused kernel produces:
# - out: [B, H*W, C] tensor ready for LayerNorm
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 768}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 1024}, num_stages=3, num_warps=8),
    ],
    key=['feat_size'],
)
@triton.jit
def fused_conv2d_add_flatten_transpose_kernel_768(
    # Input tensor [B, C, H, W]
    inp_ptr, inp_stride_b, inp_stride_c, inp_stride_h, inp_stride_w,
    # Conv weight [C, 1, 3, 3]
    weight_ptr, weight_stride_co, weight_stride_ci, weight_stride_kh, weight_stride_kw,
    # Conv bias [C]
    bias_ptr,
    # Output tensor [B, H*W, C]
    out_ptr, out_stride_b, out_stride_seq, out_stride_c,
    # Tensor dimensions
    b, c, h, w, seq_len, feat_size,
    # Conv parameters
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    # Kernel size
    kernel_size,
    # Block size for autotuning
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel: Conv2D (depthwise) + Add (residual) + Flatten + Transpose
    
    This kernel processes one [B, H*W] position per program, handling all C channels.
    """
    # Calculate position from program ID
    pid = tl.program_id(0)
    b_idx = pid // seq_len
    s_idx = pid % seq_len
    
    # Recover H, W indices from flattened sequence position
    h_idx = s_idx // w
    w_idx = s_idx % w
    
    # Bounds checking
    if b_idx >= b:
        return
    
    # Calculate output row offset for [b_idx, :, :]
    out_row_offset = b_idx * out_stride_b
    
    # Process channels in blocks for better memory access pattern
    for c_start in range(0, c, BLOCK_C):
        # Convolution accumulator for this channel block
        conv_out = tl.zeros((BLOCK_C,), dtype=tl.float32)
        
        # Depthwise convolution: each output channel corresponds to one input channel
        for kh in range(3):
            for kw in range(3):
                # Input position with padding
                inp_h = h_idx * stride_h + kh * dilation_h - padding_h
                inp_w = w_idx * stride_w + kw * dilation_w - padding_w
                
                # Bounds check for padding
                if 0 <= inp_h < h and 0 <= inp_w < w:
                    # Input pointer offsets
                    inp_h_offset = inp_h * inp_stride_h
                    inp_w_offset = inp_w * inp_stride_w
                    
                    # Load input values for all channels in block
                    inp_ptrs = inp_ptr + b_idx * inp_stride_b + (c_start + tl.arange(0, BLOCK_C)) * inp_stride_c + inp_h_offset + inp_w_offset
                    inp_vals = tl.load(inp_ptrs, mask=c_start + tl.arange(0, BLOCK_C) < c, other=0.0)
                    
                    # Load weight values [BLOCK_C, 1, 3, 3] for each channel
                    for ci in range(BLOCK_C):
                        w_offset = (c_start + ci) * weight_stride_co + 0 * weight_stride_ci + kh * weight_stride_kh + kw * weight_stride_kw
                        w_val = tl.load(weight_ptr + w_offset)
                        conv_out = conv_out + inp_vals * w_val
        
        # Add bias
        bias_vals = tl.load(bias_ptr + c_start + tl.arange(0, BLOCK_C), mask=c_start + tl.arange(0, BLOCK_C) < c, other=0.0)
        conv_out = conv_out + bias_vals
        
        # Load residual and add
        residual_ptrs = inp_ptr + b_idx * inp_stride_b + (c_start + tl.arange(0, BLOCK_C)) * inp_stride_c + h_idx * inp_stride_h + w_idx * inp_stride_w
        residual_vals = tl.load(residual_ptrs, mask=c_start + tl.arange(0, BLOCK_C) < c, other=0.0)
        result = conv_out + residual_vals
        
        # Store transposed result [B, H*W, C]
        store_ptrs = out_ptr + out_row_offset + s_idx * out_stride_seq + (c_start + tl.arange(0, BLOCK_C)) * out_stride_c
        tl.store(store_ptrs, result, mask=c_start + tl.arange(0, BLOCK_C) < c)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 768}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 1024}, num_stages=3, num_warps=8),
    ],
    key=['feat_size'],
)
@triton.jit
def fused_conv2d_add_flatten_transpose_kernel_1024(
    # Input tensor [B, C, H, W]
    inp_ptr, inp_stride_b, inp_stride_c, inp_stride_h, inp_stride_w,
    # Conv weight [C, 1, 3, 3]
    weight_ptr, weight_stride_co, weight_stride_ci, weight_stride_kh, weight_stride_kw,
    # Conv bias [C]
    bias_ptr,
    # Output tensor [B, H*W, C]
    out_ptr, out_stride_b, out_stride_seq, out_stride_c,
    # Tensor dimensions
    b, c, h, w, seq_len, feat_size,
    # Conv parameters
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    # Kernel size
    kernel_size,
    # Block size for autotuning
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel: Conv2D (depthwise) + Add (residual) + Flatten + Transpose (1024 channels)
    """
    # Calculate position from program ID
    pid = tl.program_id(0)
    b_idx = pid // seq_len
    s_idx = pid % seq_len
    
    # Recover H, W indices from flattened sequence position
    h_idx = s_idx // w
    w_idx = s_idx % w
    
    # Bounds checking
    if b_idx >= b:
        return
    
    # Calculate output row offset for [b_idx, :, :]
    out_row_offset = b_idx * out_stride_b
    
    # Process channels in blocks for better memory access pattern
    for c_start in range(0, c, BLOCK_C):
        # Convolution accumulator for this channel block
        conv_out = tl.zeros((BLOCK_C,), dtype=tl.float32)
        
        # Depthwise convolution: each output channel corresponds to one input channel
        for kh in range(3):
            for kw in range(3):
                # Input position with padding
                inp_h = h_idx * stride_h + kh * dilation_h - padding_h
                inp_w = w_idx * stride_w + kw * dilation_w - padding_w
                
                # Bounds check for padding
                if 0 <= inp_h < h and 0 <= inp_w < w:
                    # Input pointer offsets
                    inp_h_offset = inp_h * inp_stride_h
                    inp_w_offset = inp_w * inp_stride_w
                    
                    # Load input values for all channels in block
                    inp_ptrs = inp_ptr + b_idx * inp_stride_b + (c_start + tl.arange(0, BLOCK_C)) * inp_stride_c + inp_h_offset + inp_w_offset
                    inp_vals = tl.load(inp_ptrs, mask=c_start + tl.arange(0, BLOCK_C) < c, other=0.0)
                    
                    # Load weight values [BLOCK_C, 1, 3, 3] for each channel
                    for ci in range(BLOCK_C):
                        w_offset = (c_start + ci) * weight_stride_co + 0 * weight_stride_ci + kh * weight_stride_kh + kw * weight_stride_kw
                        w_val = tl.load(weight_ptr + w_offset)
                        conv_out = conv_out + inp_vals * w_val
        
        # Add bias
        bias_vals = tl.load(bias_ptr + c_start + tl.arange(0, BLOCK_C), mask=c_start + tl.arange(0, BLOCK_C) < c, other=0.0)
        conv_out = conv_out + bias_vals
        
        # Load residual and add
        residual_ptrs = inp_ptr + b_idx * inp_stride_b + (c_start + tl.arange(0, BLOCK_C)) * inp_stride_c + h_idx * inp_stride_h + w_idx * inp_stride_w
        residual_vals = tl.load(residual_ptrs, mask=c_start + tl.arange(0, BLOCK_C) < c, other=0.0)
        result = conv_out + residual_vals
        
        # Store transposed result [B, H*W, C]
        store_ptrs = out_ptr + out_row_offset + s_idx * out_stride_seq + (c_start + tl.arange(0, BLOCK_C)) * out_stride_c
        tl.store(store_ptrs, result, mask=c_start + tl.arange(0, BLOCK_C) < c)


def _fused_kernel_768(inp, weight, bias, stride, padding, dilation, groups):
    """Kernel launcher for 768 channels."""
    b, c, h, w = inp.shape
    seq_len = h * w
    
    # Prepare contiguous inputs
    inp = inp.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Output tensor [B, H*W, C]
    out = torch.empty((b, seq_len, c), dtype=inp.dtype, device=inp.device)
    
    # Calculate grid
    num_programs = b * seq_len
    grid = (num_programs,)
    
    # Launch kernel
    fused_conv2d_add_flatten_transpose_kernel_768[grid](
        inp, inp.stride(0), inp.stride(1), inp.stride(2), inp.stride(3),
        weight, weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        bias,
        out, out.stride(0), out.stride(1), out.stride(2),
        b, c, h, w, seq_len, c,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], groups,
        3,  # kernel_size
    )
    
    return out


def _fused_kernel_1024(inp, weight, bias, stride, padding, dilation, groups):
    """Kernel launcher for 1024 channels."""
    b, c, h, w = inp.shape
    seq_len = h * w
    
    # Prepare contiguous inputs
    inp = inp.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Output tensor [B, H*W, C]
    out = torch.empty((b, seq_len, c), dtype=inp.dtype, device=inp.device)
    
    # Calculate grid
    num_programs = b * seq_len
    grid = (num_programs,)
    
    # Launch kernel
    fused_conv2d_add_flatten_transpose_kernel_1024[grid](
        inp, inp.stride(0), inp.stride(1), inp.stride(2), inp.stride(3),
        weight, weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        bias,
        out, out.stride(0), out.stride(1), out.stride(2),
        b, c, h, w, seq_len, c,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], groups,
        3,  # kernel_size
    )
    
    return out


@torch.fx.wrap
def fused_conv2d_add_flatten_transpose(inp, weight, bias, stride, padding, dilation, groups, route=""):
    """
    Dispatch wrapper for the fused Conv2D + Add + Flatten + Transpose kernel.
    
    This kernel fuses:
    1. Depthwise Conv2D
    2. Element-wise addition with residual
    3. Flatten spatial dimensions
    4. Transpose to [B, H*W, C] format
    
    The output is ready for LayerNorm which will be applied by subsequent operations.
    
    Args:
        inp: Input tensor [B, C, H, W]
        weight: Conv weight [C, 1, 3, 3]
        bias: Conv bias [C]
        stride: Tuple of 2 integers
        padding: Tuple of 2 integers
        dilation: Tuple of 2 integers
        groups: Number of groups for depthwise convolution
        route: Route string to select kernel variant
        
    Returns:
        tmp_7: [B, H*W, C] - fused Conv2D+Add+Flatten+Transpose output
    """
    if route == "route_1024":
        return _fused_kernel_1024(inp, weight, bias, stride, padding, dilation, groups)
    else:
        # Default to 768 channels
        return _fused_kernel_768(inp, weight, bias, stride, padding, dilation, groups)


# ============================================================================
# Pattern Matching Function (768 channels)
# ============================================================================
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the pattern: Conv2D only
    """
    conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 768)
    return conv2d


# ============================================================================
# Replacement Arguments Function
# ============================================================================
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for the fused kernel (768 channels).
    
    in_0=LN_bias, in_1=LN_weight, in_2=conv_bias, in_3=conv_weight, in_4=input
    """
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 768
    # Route string identifies this as the 768-channel variant
    return (in_4, in_3, in_2, stride, padding, dilation, groups, "route_768")


# ============================================================================
# Replacement Function
# ============================================================================
def replacement_func():
    """
    Returns the fused kernel dispatch function.
    """
    return fused_conv2d_add_flatten_transpose