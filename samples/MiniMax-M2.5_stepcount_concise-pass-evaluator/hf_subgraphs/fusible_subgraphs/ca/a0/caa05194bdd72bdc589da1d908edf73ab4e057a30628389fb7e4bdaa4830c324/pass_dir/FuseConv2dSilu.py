import torch


def pattern(x):
    """
    Match dropout with p=0.0 and training=False (no-op).
    """
    result = torch.nn.functional.dropout(x, 0.0, False, False)
    return result


def replacement_args(x):
    return (x,)


@triton.jit
def conv2d_silu_kernel(
    # Pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Shapes
    N, C_in, H, W,
    C_out, K_h, K_w,
    # Stride info (computed)
    stride_n, stride_c_in, stride_h, stride_w,
    stride_w_o, stride_h_o, stride_c_o,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Conv2D + SiLU kernel.
    For each output element, compute convolution and apply SiLU in one pass.
    """
    # Each program computes one channel of output
    program_id_c = tl.program_id(0)
    program_id_n = tl.program_id(1)
    program_id_h = tl.program_id(2)
    program_id_w = tl.program_id(3)
    
    # Total programs = N * C_out * H_out * W_out
    # For simplicity, we'll use a flattened 1D kernel approach with better parallelization
    
    # Compute output position
    # We need N * C_out * H_out * W_out threads
    n_elements = N * C_out * H * W
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Handle multiple outputs per thread if needed
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute (n, c_out, h, w) from flat index
    n = offsets // (C_out * H * W)
    remainder = offsets % (C_out * H * W)
    c_out = remainder // (H * W)
    remainder = remainder % (H * W)
    h = remainder // W
    w = remainder % W
    
    # Convolution: sum over all input channels
    # weight shape: [C_out, C_in, K_h, K_w] = [256, 128, 1, 1]
    # For K_h=K_w=1, this is simplified
    sum_val = 0.0
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + c_out)
    
    # Iterate over input channels
    for c_in in range(C_in):
        # Load input: input[n, c_in, h, w]
        input_idx = n * stride_n + c_in * stride_c_in + h * stride_h + w * stride_w
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        
        # Load weight: weight[c_out, c_in, 0, 0]
        weight_idx = c_out * (C_in * K_h * K_w) + c_in * (K_h * K_w)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        
        sum_val += input_val * weight_val
    
    # Add bias
    conv_result = sum_val + bias
    
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_result))
    silu_result = conv_result * sigmoid_val
    
    # Store result
    output_idx = n * stride_n + c_out * stride_c_o + h * stride_h_o + w * stride_w_o
    tl.store(output_ptr + output_idx, silu_result, mask=mask)


@torch.fx.wrap
def fused_conv2d_silu_kernel(bias, weight, input):
    """
    Wrapper for fused Conv2D + SiLU kernel.
    
    Input shape: [N, C_in, H, W]
    Weight shape: [C_out, C_in, K_h, K_w]
    Bias shape: [C_out]
    Output shape: [N, C_out, H, W] (for stride=1, padding=0, dilation=1)
    """
    N, C_in, H, W = input.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Compute output shape (assuming stride=1, padding=0, dilation=1)
    H_out = H - K_h + 1
    W_out = W - K_w + 1
    
    # Allocate output
    output = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    
    # Compute strides for efficient access
    stride_n = input.stride(0)
    stride_c_in = input.stride(1)
    stride_h = input.stride(2)
    stride_w = input.stride(3)
    
    stride_n_o = output.stride(0)
    stride_c_o = output.stride(1)
    stride_h_o = output.stride(2)
    stride_w_o = output.stride(3)
    
    # Total number of output elements
    n_elements = N * C_out * H_out * W_out
    
    # Block size - tune for best performance
    BLOCK_SIZE = 1024
    
    # Grid: enough blocks to cover all elements
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # For better parallelism, use more programs
    # Use N*C_out*H_out*W_out programs with each handling BLOCK_SIZE elements
    grid = (num_programs,)
    
    conv2d_silu_kernel[grid](
        input, weight, bias, output,
        N, C_in, H, W,
        C_out, K_h, K_w,
        stride_n, stride_c_in, stride_h, stride_w,
        stride_w_o, stride_h_o, stride_c_o,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_conv2d_silu_kernel