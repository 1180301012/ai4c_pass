import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2d + Hardtanh + Mul fusion
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: Conv2d(bias, weight, input) + Hardtanh(input) * output
    """
    # Conv2d with bias (in_0), weight (in_1), input (in_2)
    # Stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Hardtanh activation with min=0.0, max=6.0
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    # Element-wise multiplication
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Optimized Triton kernel for Conv2d + Hardtanh + Mul fusion
@triton.jit
def fused_conv_hardtanh_mul_kernel(
    # Pointers
    in_ptr, weight_ptr, bias_ptr,
    hardtanh_ptr, out_ptr,
    # Strides for input [N, C_in, H, W]
    in_s0, in_s1, in_s2, in_s3,
    # Strides for weight [C_out, C_in, 1, 1]
    w_s0, w_s1, w_s2, w_s3,
    # Strides for hardtanh [N, C_out, H, W] - same as output
    h_s0, h_s1, h_s2, h_s3,
    # Strides for output [N, C_out, H, W]
    out_s0, out_s1, out_s2, out_s3,
    # Dimensions
    N, C_in, C_out, H, W,
    # Hardtanh bounds
    min_val: tl.constexpr, max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get position in output tensor
    pid = tl.program_id(0)
    n_elements = N * C_out * H * W
    
    # Bounds check
    if pid * BLOCK_SIZE >= n_elements:
        return
    
    # Calculate output indices
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode to N, C_out, H, W indices
    n = offsets // (C_out * H * W)
    tmp = offsets % (C_out * H * W)
    c_out = tmp // (H * W)
    tmp = tmp % (H * W)
    h = tmp // W
    w = tmp % W
    
    # Compute Conv2d: sum over C_in
    # weight shape: [C_out, C_in, 1, 1], so weight[c_out, c_in, 0, 0]
    # input shape: [N, C_in, H, W], so input[n, c_in, h, w]
    
    # Compute convolution (depthwise-like with 1x1 kernel)
    conv_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for c_in in range(C_in):
        # Input element: input[n, c_in, h, w]
        in_offset = n * in_s0 + c_in * in_s1 + h * in_s2 + w * in_s3
        
        # Weight element: weight[c_out, c_in, 0, 0]
        w_offset = c_out * w_s0 + c_in * w_s1
        
        # Load and multiply
        inp_val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
        w_val = tl.load(weight_ptr + w_offset, mask=mask, other=0.0)
        
        conv_sum = conv_sum + inp_val * w_val
    
    # Add bias
    bias_offset = c_out
    bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)
    conv_sum = conv_sum + bias_val
    
    # Compute Hardtanh on hardtanh input
    # hardtanh input shape: [N, C_out, H, W], same as output
    hardtanh_offset = n * h_s0 + c_out * h_s1 + h * h_s2 + w * h_s3
    hardtanh_val = tl.load(hardtanh_ptr + hardtanh_offset, mask=mask, other=0.0)
    
    # Apply hardtanh: clamp(hardtanh_val, min_val, max_val)
    hardtanh_val = tl.minimum(hardtanh_val, max_val)
    hardtanh_val = tl.maximum(hardtanh_val, min_val)
    
    # Multiply: conv_result * hardtanh_result
    result = conv_sum * hardtanh_val
    
    # Store result
    out_offset = n * out_s0 + c_out * out_s1 + h * out_s2 + w * out_s3
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_conv_hardtanh_mul_wrapper(in_0, in_1, in_2, in_3):
    """
    Fused Conv2d + Hardtanh + Mul kernel.
    
    Args:
        in_0: bias tensor [C_out]
        in_1: weight tensor [C_out, C_in, 1, 1]
        in_2: input tensor [N, C_in, H, W]
        in_3: hardtanh input tensor [N, C_out, H, W]
    
    Returns:
        Output tensor [N, C_out, H, W]
    """
    # Get shapes
    N, C_in, H, W = in_2.shape
    C_out = in_0.shape[0]
    
    # Allocate output
    out = torch.empty((N, C_out, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # Get strides (PyTorch uses NCHW format)
    in_s0, in_s1, in_s2, in_s3 = in_2.stride()
    w_s0, w_s1, w_s2, w_s3 = in_1.stride()
    h_s0, h_s1, h_s2, h_s3 = in_3.stride()
    out_s0, out_s1, out_s2, out_s3 = out.stride()
    
    # Calculate total elements
    n_elements = N * C_out * H * W
    
    # Choose block size based on problem size
    BLOCK_SIZE = 1024
    
    # Calculate grid
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv_hardtanh_mul_kernel[(num_programs,)](
        in_2, in_1, in_0,
        in_3, out,
        in_s0, in_s1, in_s2, in_s3,
        w_s0, w_s1, w_s2, w_s3,
        h_s0, h_s1, h_s2, h_s3,
        out_s0, out_s1, out_s2, out_s3,
        N, C_in, C_out, H, W,
        0.0, 6.0,  # Hardtanh bounds
        BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_conv_hardtanh_mul_wrapper