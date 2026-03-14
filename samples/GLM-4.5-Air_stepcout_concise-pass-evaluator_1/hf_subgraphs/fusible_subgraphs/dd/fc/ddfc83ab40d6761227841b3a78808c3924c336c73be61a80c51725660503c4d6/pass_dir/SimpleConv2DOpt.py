import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    """Simple conv2d pattern to debug matching"""
    result = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return result

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def simple_conv2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_out, H, W, C_in,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C_out * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape offsets to 4D: [N, C_out, H, W]
    offset_w = offsets % W
    offset_h = (offsets // W) % H
    offset_c_out = (offsets // (W * H)) % C_out
    offset_n = offsets // (W * H * C_out)
    
    # Load input tensor [N, C_in, H, W]
    # For simplicity, we assume C_in == C_out for 1x1 conv
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight tensor [C_out, C_in, 1, 1] -> squeeze to [C_out]
    weight_val = tl.load(weight_ptr + offset_c_out, mask=offset_c_out < C_out)
    bias_val = tl.load(bias_ptr + offset_c_out, mask=offset_c_out < C_out)
    
    # Apply convolution (1x1 with groups=1)
    result = x_val * weight_val + bias_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_conv2d(x, weight, bias):
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]  # For conv2d, weight shape is [C_out, C_in, kW, kH]
    
    output = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    total_elements = N * C_out * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_conv2d_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight.squeeze(),  # Remove spatial dims for 1x1 conv
        bias_ptr=bias,
        out_ptr=output,
        N=N, C_out=C_out, H=H, W=W, C_in=C_in,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return simple_conv2d