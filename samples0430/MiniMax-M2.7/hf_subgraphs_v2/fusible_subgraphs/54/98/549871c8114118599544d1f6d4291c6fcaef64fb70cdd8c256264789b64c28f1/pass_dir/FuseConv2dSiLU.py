import torch
import triton
import triton.language as tl


# Optimized 1x1 Conv2D + SiLU kernel using constexpr for compile-time optimization
@triton.jit
def fused_conv_silu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_out, C_in, H, W,
    stride_n, stride_c_in, stride_h, stride_w,
    wstride_oc, wstride_ic,
    BLOCK_W: tl.constexpr,
):
    """
    Optimized 1x1 Conv2D + SiLU kernel with constexpr for compile-time optimization.
    Grid: (C_out, H) - each block processes one output channel and all W positions.
    Uses sequential memory access for better cache utilization.
    """
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Offsets for width dimension
    offs_w = tl.arange(0, BLOCK_W)
    
    # Load bias
    bias = tl.load(bias_ptr + pid_c).to(tl.float32)
    
    # Initialize accumulator for all W positions
    acc = tl.broadcast_to(bias, (BLOCK_W,))
    
    # Loop over input channels
    for c_in in range(C_in):
        # Load weight for this input channel: scalar
        w = tl.load(weight_ptr + pid_c * wstride_oc + c_in * wstride_ic).to(tl.float32)
        
        # Load input for all W positions: [BLOCK_W]
        input_base = c_in * stride_c_in + pid_h * stride_h
        input_offsets = input_base + offs_w * stride_w
        inp = tl.load(input_ptr + input_offsets, mask=offs_w < W, other=0.0).to(tl.float32)
        
        # Multiply and accumulate (broadcast w over BLOCK_W dimension)
        acc += w * inp
    
    # Apply SiLU: x * sigmoid(x)
    result = acc * tl.sigmoid(acc)
    
    # Store output for all W positions
    output_base = pid_c * stride_c_in + pid_h * stride_h
    output_offsets = output_base + offs_w * stride_w
    tl.store(output_ptr + output_offsets, result, mask=offs_w < W)


# Autotune configuration
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 256}),
        triton.Config({'BLOCK_W': 128}),
        triton.Config({'BLOCK_W': 64}),
    ],
    key=['W'],
)
@triton.jit
def fused_conv_silu_autotuned_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_out, C_in, H, W,
    stride_n, stride_c_in, stride_h, stride_w,
    wstride_oc, wstride_ic,
    BLOCK_W: tl.constexpr,
):
    """
    Autotuned 1x1 Conv2D + SiLU kernel.
    """
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    offs_w = tl.arange(0, BLOCK_W)
    bias = tl.load(bias_ptr + pid_c).to(tl.float32)
    acc = tl.broadcast_to(bias, (BLOCK_W,))
    
    for c_in in range(C_in):
        w = tl.load(weight_ptr + pid_c * wstride_oc + c_in * wstride_ic).to(tl.float32)
        input_base = c_in * stride_c_in + pid_h * stride_h
        input_offsets = input_base + offs_w * stride_w
        inp = tl.load(input_ptr + input_offsets, mask=offs_w < W, other=0.0).to(tl.float32)
        acc += w * inp
    
    result = acc * tl.sigmoid(acc)
    output_base = pid_c * stride_c_in + pid_h * stride_h
    output_offsets = output_base + offs_w * stride_w
    tl.store(output_ptr + output_offsets, result, mask=offs_w < W)


def pattern(in_0, in_1, in_2):
    """
    Match Conv2D pattern.
    """
    result = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return result


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement.
    """
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_conv_silu_wrapper(in_0, in_1, in_2):
    """
    Fused Conv2D + SiLU wrapper using autotuned Triton kernel.
    """
    N, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]
    
    # Output tensor
    output = torch.empty((N, C_out, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # Strides
    stride_n_in, stride_c_in, stride_h_in, stride_w_in = in_2.stride()
    wstride_oc, wstride_ic, _, _ = in_1.stride()
    
    # Grid configuration: (C_out, H)
    grid = (C_out, H)
    
    fused_conv_silu_autotuned_kernel[grid](
        in_2, in_1, in_0, output,
        N, C_out, C_in, H, W,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in,
        wstride_oc, wstride_ic,
    )
    
    return output


def replacement_func():
    return fused_conv_silu_wrapper