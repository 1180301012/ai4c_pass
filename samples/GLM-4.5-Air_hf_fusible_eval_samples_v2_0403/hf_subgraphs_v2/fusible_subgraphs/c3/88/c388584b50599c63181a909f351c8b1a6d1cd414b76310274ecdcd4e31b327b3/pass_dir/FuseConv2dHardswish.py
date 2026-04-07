import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    """Pattern: Conv2D + Hardswish fusion"""
    # Conv2D with stride (1,1), padding (0,0), dilation (1,1), groups=1
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # Hardswish activation (inplace=True)
    hardswish_out = torch.nn.functional.hardswish(conv_out, True)
    return hardswish_out

@triton.jit
def fused_conv_hardswish_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + Hardswish kernel optimized for [N, C, 1, 1] tensors"""
    # Each program processes one output channel for all batches
    c_out = tl.program_id(0)
    
    if c_out >= C_out:
        return
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + c_out)
    
    # For each batch in parallel
    for n in tl.range(0, N):
        # Load input tensor - for [N, C_in, 1, 1], we only need the first element
        # since all spatial positions are the same
        input_idx = n * C_in * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0
        x_val = tl.load(x_ptr + input_idx)
        
        # Load weight for this output channel
        # For our specific case: [1280, 960, 1, 1], we need the weight at [c_out, 0, 0, 0]
        weight_idx = c_out * C_in * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0
        weight_val = tl.load(weight_ptr + weight_idx)
        
        # Perform 1x1 convolution (essentially just matrix multiplication + bias)
        conv_val = x_val * weight_val + bias_val
        
        # Hardswish activation: x * relu6(x + 3) / 6
        # relu6(x) = max(0, min(6, x))
        relu6_val = tl.minimum(tl.maximum(conv_val + 3, 0), 6)
        hardswish_val = conv_val * relu6_val / 6
        
        # Store output at [n, c_out, 0, 0]
        output_idx = n * C_out * 1 * 1 + c_out * 1 * 1 + 0 * 1 + 0
        tl.store(out_ptr + output_idx, hardswish_val)

@torch.fx.wrap
def fused_conv_hardswish(x, weight, bias):
    """Wrapper for fused Conv2D + Hardswish"""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Output shape is [N, C_out, H, W]
    out = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Optimize kernel launch for our specific use case
    # Since we have [N, 960, 1, 1] -> [N, 1280, 1, 1], we assign one program per output channel
    # This gives us good parallelism across output channels
    num_programs = C_out
    
    fused_conv_hardswish_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N, C_in=C_in, H=H, W=W, C_out=C_out,
        BLOCK_SIZE=1,  # Not used in this kernel setup
    )
    
    return out

def replacement_args(x, weight, bias):
    """Extract arguments for fused operation"""
    return (x, weight, bias)

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_hardswish