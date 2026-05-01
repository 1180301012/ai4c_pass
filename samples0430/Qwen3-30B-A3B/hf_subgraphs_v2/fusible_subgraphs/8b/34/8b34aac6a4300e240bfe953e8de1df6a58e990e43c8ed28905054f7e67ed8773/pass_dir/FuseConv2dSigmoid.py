import torch
import triton
import triton.language as tl

@triton.jit
def pointwise_conv_sigmoid_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, H, W,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_weight_c, stride_weight_d,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    batch_idx = pid_bh // (H * W)
    h_idx = (pid_bh % (H * W)) // W
    w_idx = (pid_bh % (H * W)) % W
    
    if batch_idx >= B or h_idx >= H or w_idx >= W or pid_c >= C:
        return
    
    bias_val = tl.load(bias_ptr + pid_c)
    
    acc = 0.0
    for d in range(C):
        in_val = tl.load(in_ptr + batch_idx * stride_in_b + d * stride_in_c + h_idx * stride_in_h + w_idx * stride_in_w)
        weight_val = tl.load(weight_ptr + pid_c * stride_weight_c + d * stride_weight_d)
        acc += in_val * weight_val
    
    acc += bias_val
    sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
    
    out_ptr_ = out_ptr + batch_idx * stride_out_b + pid_c * stride_out_c + h_idx * stride_out_h + w_idx * stride_out_w
    tl.store(out_ptr_, sigmoid_val)

def pattern(in_5, in_1, in_0):
    conv = torch.conv2d(in_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    sigmoid = torch.sigmoid(conv)
    return sigmoid

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@torch.fx.wrap
def conv_sigmoid_wrapper(in_5, in_1, in_0):
    B, C, H, W = in_5.shape
    
    # Get tensor strides
    stride_in_b, stride_in_c, stride_in_h, stride_in_w = in_5.stride()
    stride_weight_c, stride_weight_d, _, _ = in_1.stride()
    stride_bias = 1  # bias is 1D
    
    out = torch.empty_like(in_5)
    
    # Grid dimensions: (B * H * W, C)
    grid = (B * H * W, (C + 32 - 1) // 32)
    
    # Launch kernel
    pointwise_conv_sigmoid_kernel[grid](
        in_5, in_1, in_0, out,
        B, C, H, W,
        stride_in_b, stride_in_c, stride_in_h, stride_in_w,
        stride_weight_c, stride_weight_d,
        out.stride()[0], out.stride()[1], out.stride()[2], out.stride()[3],
        BLOCK_C=32,
        BLOCK_H=1,
        BLOCK_W=1
    )
    
    return out

def replacement_func():
    return conv_sigmoid_wrapper