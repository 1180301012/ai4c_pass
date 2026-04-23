import torch
import triton
import triton.language as tl

# Constants for GELU approximation
GELU_SCALER = 0.7978845608028654  # sqrt(2/pi)
GELU_COEFF = 0.044715


@triton.jit
def gelu_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    GELU activation kernel: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # GELU computation using tanh approximation
    # x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    inner = x + GELU_COEFF * x_cubed
    inner = inner * GELU_SCALER
    tanh_val = tl.tanh(inner)
    out = x * 0.5 * (1.0 + tanh_val)
    
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_gelu(x):
    """Pure GELU activation using Triton"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    gelu_kernel[(num_programs,)](
        output_ptr=out,
        input_ptr=x,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv2d_gelu_kernel(
    # Pointers
    output_ptr, input_ptr, weight_ptr, bias_ptr,
    # Strides
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_out_channel_stride, weight_in_channel_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    # Dimensions
    N, C_in, H_in, W_in, C_out,
    K_H, K_W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    out_h, out_w,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused Conv2D + GELU kernel.
    
    Computes: out = GELU(conv(input, weight, bias))
    
    The kernel performs:
    1. Convolution computation across the tile
    2. GELU activation applied element-wise on the result
    
    GELU formula: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    # Program IDs for 2D tile grid
    pid_batch = tl.program_id(0)
    pid_out_h = tl.program_id(1)
    pid_out_w = tl.program_id(2)
    
    # Calculate output position
    out_channel_offset = tl.arange(0, BLOCK_SIZE_M)
    out_h_offset = pid_out_h * BLOCK_SIZE_M + out_channel_offset
    out_w_offset = pid_out_w * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create output mask
    mask_out = (out_channel_offset < BLOCK_SIZE_M)[:, None] & (out_w_offset < BLOCK_SIZE_N)[None, :]
    
    # Accumulator for convolution
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load bias
    bias = tl.load(bias_ptr + out_channel_offset, mask=out_channel_offset < BLOCK_SIZE_M, other=0.0)
    
    # Calculate input window for this output position
    in_h_start = out_h_offset * stride_h - padding_h
    in_w_start = out_w_offset * stride_w - padding_w
    
    # Number of input channels to loop over (K dimension)
    K = C_in * K_H * K_W
    num_k_blocks = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    for ki in range(num_k_blocks):
        # Load weight tile: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        kw_offset = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        kw_mask = kw_offset < K
        
        c_in_offset = kw_offset // (K_H * K_W)
        k_h_offset = (kw_offset % (K_H * K_W)) // K_W
        k_w_offset = kw_offset % K_W
        
        w_ptr = (weight_ptr +
                 c_in_offset[None, :] * weight_in_channel_stride +
                 k_h_offset[None, :] * weight_height_stride +
                 k_w_offset[None, :] * weight_width_stride +
                 out_channel_offset[:, None] * weight_out_channel_stride)
        w = tl.load(w_ptr, mask=(kw_mask[None, :] & out_channel_offset[:, None] < BLOCK_SIZE_M), other=0.0)
        
        # Load input tile: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        ih_base = in_h_start[:, None] + k_h_offset[None, :]
        iw_base = in_w_start[:, None] + k_w_offset[None, :]
        
        # Compute input offsets
        in_h_offsets = ih_base
        in_w_offsets = iw_base
        
        # Valid input positions (within bounds)
        valid_h = (in_h_offsets >= 0) & (in_h_offsets < H_in)
        valid_w = (in_w_offsets >= 0) & (in_w_offsets < W_in)
        valid_mask = valid_h & valid_w
        
        # Compute flat offsets for input
        batch_idx = pid_batch
        in_flat_offsets = (batch_idx * input_batch_stride +
                          c_in_offset[None, :] * input_channel_stride +
                          in_h_offsets * input_height_stride +
                          in_w_offsets * input_width_stride)
        
        x = tl.load(input_ptr + in_flat_offsets, mask=(kw_mask[None, :] & valid_mask & out_channel_offset[:, None] < BLOCK_SIZE_M), other=0.0)
        
        # Perform dot product
        acc += tl.sum(w * x, axis=1)
    
    # Add bias
    acc = acc + bias[:, None]
    
    # Apply GELU: x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    acc_f32 = acc.to(tl.float32)
    x_cubed = acc_f32 * acc_f32 * acc_f32
    inner = acc_f32 + GELU_COEFF * x_cubed
    inner = inner * GELU_SCALER
    tanh_val = tl.tanh(inner)
    gelu_out = acc_f32 * 0.5 * (1.0 + tanh_val)
    
    # Store output
    out_base = (pid_batch * output_batch_stride +
                out_channel_offset[:, None] * output_channel_stride +
                out_h_offset[:, None] * output_height_stride +
                out_w_offset[:, None] * output_width_stride)
    
    tl.store(output_ptr + out_base, gelu_out.to(tl.float16), mask=mask_out)


def pattern(in_0, in_1, in_2):
    """
    Pattern: Conv2D with bias followed by GELU activation
    Note: Dropout with p=0.0 is a no-op and not included in the pattern
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    # For Conv2D: in_0=bias, in_1=weight, in_2=input
    return (in_0, in_1, in_2, 128)


def replacement_func():
    return fused_conv2d_gelu_128


@torch.fx.wrap
def fused_conv2d_gelu_128(bias, weight, input, groups):
    """
    Fused Conv2D + GELU kernel for out_channels=128
    """
    N, C_in, H_in, W_in = input.shape
    C_out = bias.shape[0]
    K_H, K_W = weight.shape[2], weight.shape[3]
    
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    # Calculate output dimensions
    out_h = (H_in + 2 * padding_h - dilation_h * (K_H - 1) - 1) // stride_h + 1
    out_w = (W_in + 2 * padding_w - dilation_w * (K_W - 1) - 1) // stride_w + 1
    
    # Allocate output
    output = torch.empty((N, C_out, out_h, out_w), dtype=input.dtype, device=input.device)
    
    # Grid: (N, out_h, out_w)
    # BLOCK_SIZE_M = C_out (output channels), BLOCK_SIZE_N = 1 (spatial positions per program)
    # Use large tile for channels
    grid = (N, out_h, out_w)
    
    # Strides
    input_batch_stride = input.stride(0)
    input_channel_stride = input.stride(1)
    input_height_stride = input.stride(2)
    input_width_stride = input.stride(3)
    
    weight_out_channel_stride = weight.stride(0)
    weight_in_channel_stride = weight.stride(1)
    weight_height_stride = weight.stride(2)
    weight_width_stride = weight.stride(3)
    
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_height_stride = output.stride(2)
    output_width_stride = output.stride(3)
    
    # Launch kernel - use BLOCK_SIZE_M = min(128, C_out), BLOCK_SIZE_N = 1 for this spatial tiling
    BLOCK_SIZE_M = min(128, C_out)
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 32
    
    fused_conv2d_gelu_kernel[grid](
        output, input, weight, bias,
        input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
        weight_out_channel_stride, weight_in_channel_stride, weight_height_stride, weight_width_stride,
        output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
        N, C_in, H_in, W_in, C_out,
        K_H, K_W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        out_h, out_w,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return output