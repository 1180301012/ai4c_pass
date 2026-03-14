import torch
import triton
import triton.language as tl


# Simple pattern - just match Conv2d
def pattern(in_0, in_1, in_2):
    out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Optimized fused Conv2d + ReLU kernel using Triton
@triton.jit
def fused_conv2d_relu_kernel(
    # Input pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Input dimensions
    N, C_in, H_in, W_in,
    # Weight dimensions
    C_out, C_in_kernel, H_kernel, W_kernel,
    # Output dimensions
    H_out, W_out,
    # Strides
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_co, weight_stride_ci, weight_stride_hk, weight_stride_wk,
    output_stride_n, output_stride_co, output_stride_h, output_stride_w,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate total number of output elements
    num_outputs = N * C_out * H_out * W_out
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_outputs
    
    # Calculate output indices
    n = offs // (C_out * H_out * W_out)
    rem = offs % (C_out * H_out * W_out)
    co = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    ho = rem // W_out
    wo = rem % W_out
    
    # Compute output index for loading
    output_idx = n * output_stride_n + co * output_stride_co + ho * output_stride_h + wo * output_stride_w
    
    # Initialize accumulator with proper type for Triton
    accumulator = tl.zeros([BLOCK_SIZE], tl.float32)
    
    for ci in range(C_in):
        for hk in range(H_kernel):
            for wk in range(W_kernel):
                # Input indices
                hi = ho + hk  # assuming stride=1, padding=0
                wi = wo + wk
                
                # Create bounds mask
                bounds_mask = (hi < H_in) & (wi < W_in)
                load_mask = mask & bounds_mask
                
                # Input index
                input_idx = n * input_stride_n + ci * input_stride_c + hi * input_stride_h + wi * input_stride_w
                # Weight index
                weight_idx = co * weight_stride_co + ci * weight_stride_ci + hk * weight_stride_hk + wk * weight_stride_wk
                
                # Load values with masking
                input_val = tl.load(input_ptr + input_idx, mask=load_mask, other=0.0)
                weight_val = tl.load(weight_ptr + weight_idx, mask=load_mask, other=0.0)
                
                # Accumulate
                accumulator = accumulator + (input_val * weight_val)
    
    # Add bias
    bias_val = tl.load(bias_ptr + co)
    accumulator = accumulator + bias_val
    
    # ReLU activation
    accumulator = tl.maximum(accumulator, 0.0)
    
    # Store result
    tl.store(output_ptr + output_idx, accumulator, mask=mask)


def fused_conv2d_relu_impl(bias, weight, input):
    # Ensure inputs are on GPU
    if input.device.type == 'cpu':
        input = input.cuda()
    if weight.device.type == 'cpu':
        weight = weight.cuda()
    if bias.device.type == 'cpu':
        bias = bias.cuda()
    
    # Get dimensions
    N, C_in, H_in, W_in = input.shape
    C_out, C_in_k, H_kernel, W_kernel = weight.shape
    
    # Output dimensions (stride=1, padding=0)
    H_out = H_in - H_kernel + 1
    W_out = W_in - W_kernel + 1
    
    # Create output tensor
    output = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    
    # Calculate total elements
    num_elements = N * C_out * H_out * W_out
    
    # Configure block size
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Get strides
    input_stride_n, input_stride_c, input_stride_h, input_stride_w = input.stride()
    weight_stride_co, weight_stride_ci, weight_stride_hk, weight_stride_wk = weight.stride()
    output_stride_n, output_stride_co, output_stride_h, output_stride_w = output.stride()
    
    # Launch kernel
    fused_conv2d_relu_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, C_in_kernel=C_in_k, H_kernel=H_kernel, W_kernel=W_kernel,
        H_out=H_out, W_out=W_out,
        input_stride_n=input_stride_n, input_stride_c=input_stride_c,
        input_stride_h=input_stride_h, input_stride_w=input_stride_w,
        weight_stride_co=weight_stride_co, weight_stride_ci=weight_stride_ci,
        weight_stride_hk=weight_stride_hk, weight_stride_wk=weight_stride_wk,
        output_stride_n=output_stride_n, output_stride_co=output_stride_co,
        output_stride_h=output_stride_h, output_stride_w=output_stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def kernel_wrapper(bias, weight, input):
    return fused_conv2d_relu_impl(bias, weight, input)


def replacement_func():
    return kernel_wrapper