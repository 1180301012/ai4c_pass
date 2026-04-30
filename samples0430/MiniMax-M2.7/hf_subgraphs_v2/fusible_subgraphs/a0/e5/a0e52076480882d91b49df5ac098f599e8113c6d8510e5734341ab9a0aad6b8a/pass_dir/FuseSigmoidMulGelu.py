import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_mul_gelu_kernel(
    conv_ptr, in2_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    dtype_val: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Fused sigmoid + mul + gelu kernel with optimized parallelization.
    
    Inputs:
        conv: [B, C, 1, 1] - conv output (attention scores)
        in2:  [B, C, H, W] - feature map to scale
    
    Output:
        out: [B, C, H, W] - scaled features after gelu
    """
    # Get program ID - each program handles multiple elements
    pid = tl.program_id(0)
    
    # Calculate total elements
    HW = H * W
    
    # Stride calculations for [B, C, H, W] tensor
    stride_b = C * HW
    stride_c = HW
    stride_h = W
    
    # Calculate which batch and channel this program handles
    batch = pid // C
    ch = pid % C
    
    # Load sigmoid value for this batch, channel
    conv_offset = batch * C + ch
    conv_val = tl.load(conv_ptr + conv_offset).to(tl.float32)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    # Process all HW elements - use vectorized loading
    for start in range(0, HW, BLOCK_SIZE):
        # Calculate offsets for this block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        
        # Calculate actual memory offsets
        # offsets // W gives h, offsets % W gives w
        h_offsets = (offsets // W) * stride_h
        w_offsets = offsets % W
        mem_offsets = batch * stride_b + ch * stride_c + h_offsets + w_offsets
        
        # Load in2 values
        in2_val = tl.load(in2_ptr + mem_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Multiply with sigmoid (broadcast)
        mul_val = sigmoid_val * in2_val
        
        # Apply GELU approximation
        x_cubed = mul_val * mul_val * mul_val
        gelu_arg = 0.7978845608028654 * (mul_val + 0.044715 * x_cubed)
        
        # Numerically stable tanh
        gelu_arg_clipped = tl.where(gelu_arg > 10.0, 10.0, tl.where(gelu_arg < -10.0, -10.0, gelu_arg))
        exp_2x = tl.exp(2.0 * gelu_arg_clipped)
        tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
        
        # GELU result
        gelu_val = 0.5 * mul_val * (1.0 + tanh_val)
        
        # Store result
        if dtype_val == 0:
            tl.store(out_ptr + mem_offsets, gelu_val, mask=mask)
        elif dtype_val == 1:
            tl.store(out_ptr + mem_offsets, gelu_val.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + mem_offsets, gelu_val.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def sigmoid_mul_gelu_wrapper(conv_output, in_2):
    """
    Fused sigmoid + multiply + gelu operation.
    """
    B, C, _, _ = conv_output.shape
    _, _, H, W = in_2.shape
    
    out = torch.empty_like(in_2)
    
    # Determine dtype encoding for kernel
    dtype_val = 0
    if in_2.dtype == torch.float16:
        dtype_val = 1
    elif in_2.dtype == torch.bfloat16:
        dtype_val = 2
    
    grid = (B * C,)
    BLOCK_SIZE = 1024
    
    sigmoid_mul_gelu_kernel[grid](
        conv_output, in_2, out,
        B, C, H, W,
        dtype_val, BLOCK_SIZE
    )
    
    return out


def pattern(conv_output, in_2):
    """
    Pattern: sigmoid(conv_output) * in_2, then gelu
    """
    sigmoid_out = conv_output.sigmoid()
    mul_out = in_2 * sigmoid_out
    gelu_out = torch.nn.functional.gelu(mul_out, approximate='none')
    return gelu_out


def replacement_args(conv_output, in_2):
    return (conv_output, in_2)


def replacement_func():
    return sigmoid_mul_gelu_wrapper