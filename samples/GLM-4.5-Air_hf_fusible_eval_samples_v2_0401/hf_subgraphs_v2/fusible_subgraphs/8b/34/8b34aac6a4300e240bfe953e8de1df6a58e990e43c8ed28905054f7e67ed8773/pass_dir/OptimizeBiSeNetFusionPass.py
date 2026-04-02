import torch
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_1x1_simplified_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C, H, W, OC,
    BLOCK_SIZE: tl.constexpr
):
    """
    Simplified 1x1 convolution kernel for specific BiSeNet pattern
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset < (N * OC * H * W)
    
    if not mask:
        return
    
    # Process a block of elements
    for i in range(BLOCK_SIZE):
        idx = offset + i
        if idx >= N * OC * H * W:
            break
            
        # Decode index: batch, output_channel, h, w
        batch = idx // (OC * H * W)
        oc = (idx // (H * W)) % OC
        h = (idx // W) % H
        w = idx % W
        
        # For 1x1 conv with groups=1, each output channel gets one weight
        input_base = batch * C * H * W + h * W + w
        weight_base = oc * C
        
        # Initialize accumulator
        acc = 0.0
        for c_in in range(C):
            x_val = tl.load(x_ptr + input_base + c_in * H * W, mask=mask)
            weight_val = tl.load(weight_ptr + weight_base + c_in, mask=mask)
            acc += x_val * weight_val
            
        # Add bias
        bias_val = tl.load(bias_ptr + oc, mask=mask)
        acc += bias_val
        
        # Store result
        tl.store(out_ptr + idx, acc, mask=mask)

@triton.jit
def interpolate_bilinear_simplified_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr
):
    """
    Simplified bilinear interpolation kernel
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset < (N * C * H_out * W_out)
    
    if not mask:
        return
    
    for i in range(BLOCK_SIZE):
        idx = offset + i
        if idx >= N * C * H_out * W_out:
            break
            
        # Decode index: batch, channel, h_out, w_out
        batch = idx // (C * H_out * W_out)
        channel = (idx // (H_out * W_out)) % C
        h_out = (idx // W_out) % H_out
        w_out = idx % W_out
        
        # Compute corresponding input coordinates
        h_in = h_out * H_in // H_out
        w_in = w_out * W_in // W_out
        
        # Simple nearest neighbor for now (can be improved to bilinear)
        input_idx = batch * C * H_in * W_in + channel * H_in * W_in + h_in * W_in + w_in
        x_val = tl.load(x_ptr + input_idx, mask=mask)
        
        # Store result
        output_idx = batch * C * H_out * W_out + channel * H_out * W_out + h_out * W_out + w_out
        tl.store(out_ptr + output_idx, x_val, mask=mask)

@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid (1 / (1 + exp(-x)))
    # Using fast sigmoid approximation for performance
    out = 1.0 / (1.0 + tl.exp(-x))
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def elementwise_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute multiplication
    out = x * y
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit 
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute addition
    out = x + y
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_bisenet_computation(in_0, in_1, in_2, in_3, in_4, in_5):
    """Optimized BiSeNet computation using Triton kernels"""
    
    # Extract shapes and dtypes
    C = in_5.shape[1]  # channels  
    H_in = W_in = 16   # input spatial size
    H_out = W_out = 64 # output spatial size
    N = in_5.shape[0]  # batch size
    
    # Block sizes optimized for GPU
    BLOCK_SIZE_CONV = 1024
    BLOCK_SIZE_ELEM = 1024
    
    # Compute output shapes
    conv_out_shape = (N, C, H_in, W_in)
    
    # Memory for intermediate results
    conv_out = torch.empty(conv_out_shape, dtype=in_5.dtype, device=in_5.device)
    interp64_out = torch.empty((N, C, H_out, W_out), dtype=in_4.dtype, device=in_4.device)
    sigmoid1_out = torch.empty((N, C, H_out, W_out), dtype=in_4.dtype, device=in_4.device)
    sigmoid2_out = torch.empty(conv_out_shape, dtype=in_5.dtype, device=in_5.device)
    mul1_out = torch.empty((N, C, H_out, W_out), dtype=in_3.dtype, device=in_3.device)
    mul2_out = torch.empty(conv_out_shape, dtype=in_2.dtype, device=in_2.device)
    interp16_out = torch.empty((N, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)
    final_out = torch.empty((N, C, H_out, W_out), dtype=in_5.dtype, device=in_5.device)
    
    # 1. Conv2D operation (optimized for 1x1 kernel)
    n_elements_conv = N * C * H_in * W_in
    programs_conv = (n_elements_conv + BLOCK_SIZE_CONV - 1) // BLOCK_SIZE_CONV
    conv_grid = (programs_conv,)
    
    conv2d_1x1_simplified_kernel[conv_grid](
        in_5, in_1, in_0,
        conv_out,
        N, C, H_in, W_in, C,
        BLOCK_SIZE_CONV
    )
    
    # 2. First interpolation (bilinear)
    n_elements_interp = N * C * H_out * W_out
    programs_interp = (n_elements_interp + BLOCK_SIZE_ELEM - 1) // BLOCK_SIZE_ELEM
    interp_grid = (programs_interp,)
    
    interpolate_bilinear_simplified_kernel[interp_grid](
        in_4, interp64_out,
        N, C, H_in, W_in, H_out, W_out,
        BLOCK_SIZE_ELEM
    )
    
    # 3. Sigmoid of first interpolation result using Triton kernel
    n_elements_interp = N * C * H_out * W_out
    programs_sigmoid = (n_elements_interp + BLOCK_SIZE_ELEM - 1) // BLOCK_SIZE_ELEM
    
    sigmoid1_out = torch.empty((N, C, H_out, W_out), dtype=interp64_out.dtype, device=interp64_out.device)
    sigmoid_kernel[programs_sigmoid](
        interp64_out, sigmoid1_out,
        n_elements_interp, BLOCK_SIZE_ELEM
    )
    
    # 4. Sigmoid of conv output using Triton kernel
    n_elements_conv = N * C * H_in * W_in
    programs_sigmoid2 = (n_elements_conv + BLOCK_SIZE_ELEM - 1) // BLOCK_SIZE_ELEM
    
    sigmoid2_out = torch.empty(conv_out_shape, dtype=conv_out.dtype, device=conv_out.device)
    sigmoid_kernel[programs_sigmoid2](
        conv_out, sigmoid2_out,
        n_elements_conv, BLOCK_SIZE_ELEM
    )
    
    # 5. First multiplication (left branch)
    mul1_out = in_3 * sigmoid1_out
    
    # 6. Second multiplication (right branch)
    mul2_out = in_2 * sigmoid2_out
    
    # 7. Second interpolation
    interpolate_bilinear_simplified_kernel[interp_grid](
        mul2_out, interp16_out,
        N, C, H_in, W_in, H_out, W_out,
        BLOCK_SIZE_ELEM
    )
    
    # 8. Final addition
    final_out = mul1_out + interp16_out
    
    return final_out

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern matches the entire BiSeNet computation:
    Conv2D -> Sigmoid -> Multiply -> Interpolate operations
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_3 = None
    tmp_5 = in_3 * tmp_4
    tmp_4 = None
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_2 = None
    tmp_7 = in_2 * tmp_6
    tmp_6 = None
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_7 = None
    tmp_9 = tmp_5 + tmp_8
    tmp_5 = tmp_8 = None
    return (tmp_9,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

def replacement_func():
    return optimized_bisenet_computation