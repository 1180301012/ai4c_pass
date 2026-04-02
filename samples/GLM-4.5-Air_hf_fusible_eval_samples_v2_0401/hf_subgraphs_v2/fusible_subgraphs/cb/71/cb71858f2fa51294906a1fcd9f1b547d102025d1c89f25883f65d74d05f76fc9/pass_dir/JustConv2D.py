import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def simple_conv2d_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, out_c, in_c, in_h, in_w, out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_elems = batch * out_c * out_h * out_w
    mask = pid < total_elems
    
    # Map to output coordinates [batch, out_c, out_h, out_w]
    b = pid // (out_c * out_h * out_w)
    rem = pid % (out_c * out_h * out_w)
    c = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    h = rem2 // out_w
    w = rem2 % out_w
    
    # Compute convolution result
    conv_val = 0.0
    
    for ic in range(in_c):
        for kh in range(in_h):
            for kw in range(in_w):
                # Load weight: [out_c, in_c, kh, kw]
                weight_offset = c * in_c * in_h * in_w + ic * in_h * in_w + kh * in_w + kw
                weight = tl.load(weight_ptr + weight_offset, 
                               mask=(c < out_c) & (ic < in_c), 
                               other=0.0)
                
                # Load input: [batch, in_c, in_h, in_w]
                input_offset = b * in_c * in_h * in_w + ic * in_h * in_w + kh * in_w + w
                input_val = tl.load(x_ptr + input_offset, 
                                  mask=(b < batch) & (ic < in_c), 
                                  other=0.0)
                
                conv_val += weight * input_val
    
    # Add bias
    bias = tl.load(bias_ptr + c, mask=(c < out_c), other=0.0)
    conv_val += bias
    
    # Store result in [batch, out_c, out_h, out_w] format
    tl.store(out_ptr + pid, conv_val, mask=mask)

@torch.fx.wrap
def simple_conv2d_func(in_0, in_1, in_2):
    batch, in_c, in_h, in_w = in_2.shape
    out_c = in_1.shape[0]
    out_h = in_h  # Same due to padding=0, stride=1  
    out_w = in_w
    
    output = torch.empty((batch, out_c, out_h, out_w), dtype=in_2.dtype, device=in_2.device)
    
    in_1 = in_1.to(in_2.device)
    in_0 = in_0.to(in_2.device)
    
    total_elems = batch * out_c * out_h * out_w
    BLOCK_SIZE = 1024
    grid_size = (triton.cdiv(total_elems, BLOCK_SIZE),)
    
    simple_conv2d_kernel[grid_size](
        in_2, in_1, in_0, output,
        batch, out_c, in_c, in_h, in_w, out_h, out_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return simple_conv2d_func