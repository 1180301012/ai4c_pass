import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_linear_permute_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride0, input_stride1, input_stride2, input_stride3,
    weight_stride0, weight_stride1,
    bias_stride0,
    output_stride0, output_stride1, output_stride2, output_stride3,
    num_h, num_w, num_channels, num_c_in,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    bx = tl.program_id(0)
    by = tl.program_id(1)
    
    h_start = bx * BLOCK_H
    w_start = by * BLOCK_W
    
    h = h_start + tl.arange(0, BLOCK_H)
    w = w_start + tl.arange(0, BLOCK_W)
    h = tl.where(h < num_h, h, 0)
    w = tl.where(w < num_w, w, 0)
    
    # Load input for current block [BLOCK_H, BLOCK_W, 3]
    input_data = tl.load(
        input_ptr + 
        h[:, None] * input_stride1 + 
        w[None, :] * input_stride2 + 
        tl.arange(0, num_c_in) * input_stride3,
        mask=(h[:, None] < num_h) & (w[None, :] < num_w),
        other=0.0
    )
    
    # Load weight [16, 3]
    weight = tl.load(
        weight_ptr + 
        tl.arange(0, num_channels)[:, None] * weight_stride0 + 
        tl.arange(0, num_c_in)[None, :] * weight_stride1,
        mask=(tl.arange(0, num_channels)[:, None] < num_channels),
        other=0.0
    )
    
    # Compute dot product: input_data @ weight.T -> [BLOCK_H, BLOCK_W, 16]
    output = tl.dot(input_data, weight.T)
    
    # Add bias
    bias = tl.load(
        bias_ptr + tl.arange(0, num_channels) * bias_stride0,
        mask=(tl.arange(0, num_channels) < num_channels),
        other=0.0
    )
    output = output + bias[None, None, :]
    
    # Store output [1, 16, 196, 196]
    output_ptrs = output_ptr + \
        tl.arange(0, num_channels)[:, None, None] * output_stride1 + \
        h[:, None, None] * output_stride2 + \
        w[None, :, None] * output_stride3
    tl.store(output_ptrs, output, mask=(h[:, None, None] < num_h) & (w[None, :, None] < num_w))

@torch.fx.wrap
def fused_linear_permute(x, weight, bias):
    B, H, W, C_in = x.shape
    C_out = weight.shape[0]
    
    output = torch.empty(B, C_out, H, W, dtype=x.dtype, device=x.device)
    
    # Get strides
    input_stride0, input_stride1, input_stride2, input_stride3 = x.stride()
    output_stride0, output_stride1, output_stride2, output_stride3 = output.stride()
    weight_stride0, weight_stride1 = weight.stride()
    bias_stride0 = bias.stride()[0]
    
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid_h = (H + BLOCK_H - 1) // BLOCK_H
    grid_w = (W + BLOCK_W - 1) // BLOCK_W
    
    fused_linear_permute_kernel[(grid_h, grid_w)](
        x, weight, bias, output,
        input_stride0, input_stride1, input_stride2, input_stride3,
        weight_stride0, weight_stride1,
        bias_stride0,
        output_stride0, output_stride1, output_stride2, output_stride3,
        H, W, C_out, C_in,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return output

def replacement_func():
    return fused_linear_permute