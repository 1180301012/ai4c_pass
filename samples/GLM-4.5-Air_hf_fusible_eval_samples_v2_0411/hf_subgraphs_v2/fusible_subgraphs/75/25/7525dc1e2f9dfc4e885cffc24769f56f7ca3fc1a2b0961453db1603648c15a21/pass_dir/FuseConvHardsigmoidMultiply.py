import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """
    Match Conv2D + Hardsigmoid + Element-wise Multiply pattern
    """
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    tmp_4 = in_2 * tmp_3
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused Conv2D + Hardsigmoid + Multiply
@triton.jit
def fused_conv_hardsigmoid_multiply_kernel(
    # Input pointers
    in_3_ptr,  # [N, C_in, H_in, W_in] 
    weight_ptr,  # [C_out, C_in, 1, 1]
    bias_ptr,  # [C_out]
    # Feature map multiplier pointer  
    mult_ptr,  # [N, C_out, H_out, W_out]
    # Output pointer
    out_ptr,  # [N, C_out, H_out, W_out]
    # Tensor shapes
    N, C_out, C_in, H_in, W_in, H_out, W_out,
    # Strides
    in_3_stride_N, in_3_stride_C, in_3_stride_H, in_3_stride_W,
    weight_stride_C_out, weight_stride_C_in, weight_stride_0, weight_stride_1,
    bias_stride_0,
    mult_stride_N, mult_stride_C, mult_stride_H, mult_stride_W,
    out_stride_N, out_stride_C, out_stride_H, out_stride_W,
    # Parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Determine program position
    pid = tl.program_id(0)
    # Calculate which output position this program handles
    c_pid = pid // (H_out * W_out)  # channel index
    h_pid = (pid % (H_out * W_out)) // W_out  # height index  
    w_pid = pid % W_out  # width index
    
    # Guard against out-of-bound accesses
    if c_pid >= C_out or h_pid >= H_out or w_pid >= W_out:
        return
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + c_pid * bias_stride_0)
    
    # Initialize output with bias
    out_val = bias_val
    
    # Loop over input channels to perform convolution
    for c_in_idx in range(0, C_in, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, C_in - c_in_idx)
        
        # Calculate input channel range for this block
        c_in_start = c_in_idx
        c_in_end = c_in_idx + block_size
        
        # Load weight vector for this output channel
        weight_ptr_base = weight_ptr + c_pid * (weight_stride_C_out + weight_stride_C_in + weight_stride_0 + weight_stride_1)
        weight_block = tl.load(weight_ptr_base + c_in_idx * weight_stride_C_in)
        
        # Load input element
        in_3_ptr_base = in_3_ptr + 0 * (in_3_stride_N) + 0 * (in_3_stride_C) + h_pid * (in_3_stride_H) + w_pid * (in_3_stride_W)
        in_3_val = tl.load(in_3_ptr_base + c_in_idx * in_3_stride_C)
        
        # Convolution computation: weight * input
        conv_val = weight_block * in_3_val
        out_val += conv_val
    
    # Apply hardsigmoid: max(0, min(1, x*(3 + 3) + 3) / 6)
    # Hardsigmoid formula: max(0, min(1, x * 3 + 3)) / 6
    hardsigmoid_val = tl.maximum(0.0, tl.minimum(1.0, out_val * 3.0 + 3.0)) / 6.0
    
    # Load multiplier value (broadcast from channel position)
    mult_ptr_base = mult_ptr + 0 * mult_stride_N + c_pid * mult_stride_C + h_pid * mult_stride_H + w_pid * mult_stride_W
    mult_val = tl.load(mult_ptr_base)
    
    # Multiplication
    out_final = hardsigmoid_val * mult_val
    
    # Store result
    out_ptr_base = out_ptr + 0 * out_stride_N + c_pid * out_stride_C + h_pid * out_stride_H + w_pid * out_stride_W
    tl.store(out_ptr_base, out_final)

# Kernel wrapper
@torch.fx.wrap
def fused_conv_hardsigmoid_multiply(in_0, in_1, in_2, in_3):
    # Get tensor shapes
    N, C_in, H_in, W_in = in_3.shape
    C_out = in_1.shape[0]  # Output channels from weight shape [C_out, C_in, 1, 1]
    H_out, W_out = H_in, W_in  # 1x1 conv preserves spatial dims
    
    # Get tensor strides
    in_3_stride_N, in_3_stride_C, in_3_stride_H, in_3_stride_W = in_3.stride()
    weight_stride_C_out, weight_stride_C_in, weight_stride_0, weight_stride_1 = in_1.stride()
    bias_stride_0 = in_0.stride(0) if len(in_0.stride()) == 1 else in_0.stride()[0]
    mult_stride_N, mult_stride_C, mult_stride_H, mult_stride_W = in_2.stride()
    out_stride_N, out_stride_C, out_stride_H, out_stride_W = in_2.stride()
    
    # Create output tensor with same shape as in_2
    out = torch.empty_like(in_2)
    
    # Calculate grid size
    total_elements = C_out * H_out * W_out
    BLOCK_SIZE = 32  # Number of input channels to process per program
    grid_size = (triton.cdiv(total_elements, 1),)
    
    # Ensure we launch enough programs to cover all output positions
    if grid_size[0] == 0:
        grid_size = (1,)
    
    # Launch kernel
    fused_conv_hardsigmoid_multiply_kernel[grid_size](
        in_3_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        mult_ptr=in_2,
        out_ptr=out,
        N=N, C_out=C_out, C_in=C_in, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        in_3_stride_N=in_3_stride_N, in_3_stride_C=in_3_stride_C, in_3_stride_H=in_3_stride_H, in_3_stride_W=in_3_stride_W,
        weight_stride_C_out=weight_stride_C_out, weight_stride_C_in=weight_stride_C_in, weight_stride_0=weight_stride_0, weight_stride_1=weight_stride_1,
        bias_stride_0=bias_stride_0,
        mult_stride_N=mult_stride_N, mult_stride_C=mult_stride_C, mult_stride_H=mult_stride_H, mult_stride_W=mult_stride_W,
        out_stride_N=out_stride_N, out_stride_C=out_stride_C, out_stride_H=out_stride_H, out_stride_W=out_stride_W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_conv_hardsigmoid_multiply