import torch
import triton
import triton.language as tl

# Pattern matching for Conv2D + addition fusion  
def pattern(in_6, in_5, weight_in_0):
    # Match the exact computation from the original model:
    # tmp_5 = torch.conv2d(in_6, weight_in_0, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1)
    # tmp_6 = in_5 + tmp_5
    tmp_5 = torch.conv2d(in_6, weight_in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = in_5 + tmp_5
    return tmp_5, tmp_6

def replacement_args(in_6, in_5, weight_in_0):
    return (in_6, in_5, weight_in_0)

# Triton kernel for fused Conv2D + Addition (1x1 convolution with residual addition)
@triton.jit
def fused_conv_add_kernel(
    x_ptr,  # in_6: [N, 512, H, W]  
    weight_ptr,  # weight: [128, 512, 1, 1]
    y_ptr,  # in_5: [N, 128, H, W]
    out_ptr,  # output: [N, 128, H, W]
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one spatial location (H x W)
    h = pid // W
    w = pid % W
    
    if h >= H or w >= W:
        return
    
    # Weight is [C_out, C_in] for matrix multiply
    weight = tl.load(weight_ptr)
    
    # Load input slice [N, C_in] for current spatial location
    x_offset = h * W * C_in + w * C_in
    y_offset = h * W * C_out + w * C_out
    
    x_slice = tl.load(x_ptr + x_offset + tl.arange(0, C_in)[None, :], 
                     mask=tl.arange(0, C_in)[None, :] < C_in)
    y_slice = tl.load(y_ptr + y_offset + tl.arange(0, C_out)[None, :], 
                     mask=tl.arange(0, C_out)[None, :] < C_out)
    
    # Matrix multiply: [N, C_in] x [C_in, C_out] -> [N, C_out]
    conv_result = tl.zeros((N, C_out), dtype=tl.float32)
    for k in range(C_in):
        conv_result += x_slice[:, k][None, :] * weight[None, :]
    
    # Add residual
    out_slice = conv_result + y_slice
    
    # Store result
    out_ptr_base = out_ptr + y_offset
    tl.store(out_ptr_base + tl.arange(0, C_out)[None, :], out_slice, 
             mask=tl.arange(0, C_out)[None, :] < C_out)

@torch.fx.wrap  
def fused_conv_add_triton(x, y, weight):
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Allocate output
    out = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Launch kernel - each spatial location has one program
    grid = H * W
    fused_conv_add_kernel[grid](
        x_ptr=x,
        weight_ptr=weight, 
        y_ptr=y,
        out_ptr=out,
        N=N,
        C_in=C_in,
        C_out=C_out, 
        H=H,
        W=W,
    )
    
    return None, out  # Return None (tmp_5), out (tmp_6) to match pattern

def replacement_func():
    return fused_conv_add_triton