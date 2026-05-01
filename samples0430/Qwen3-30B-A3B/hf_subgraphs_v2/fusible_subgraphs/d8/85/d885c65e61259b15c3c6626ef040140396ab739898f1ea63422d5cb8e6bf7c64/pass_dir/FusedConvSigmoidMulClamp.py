import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp3 = conv.sigmoid()
    tmp4 = in_2 * tmp3
    tmp5 = torch.nn.functional.hardtanh(tmp4, 0.0, 6.0, False)
    return tmp5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv_sigmoid_mul_clamp_kernel(
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    in_2_ptr,
    out_ptr,
    B, C_out, H, W, in_C,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    c_out = tl.thread_id(0)
    mask = c_out < C_out

    in_3_offset = batch_idx * in_C * 1 * 1
    in_3_load = tl.load(in_3_ptr + in_3_offset, mask=mask & (tl.arange(0, in_C) < in_C), other=0.0)
    
    weight_offset = c_out * in_C
    weight_load = tl.load(in_1_ptr + weight_offset, mask=mask & (tl.arange(0, in_C) < in_C), other=0.0)
    
    conv_val = tl.dot(in_3_load, weight_load)
    bias_load = tl.load(in_0_ptr + c_out, mask=mask)
    conv_val += bias_load
    
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    in_2_offset = batch_idx * C_out * H * W + c_out * H * W + h * W + w
    scale_val = tl.load(in_2_ptr + in_2_offset, mask=mask)
    mul_val = sigmoid_val * scale_val
    
    clamped_val = tl.minimum(tl.maximum(mul_val, 0.0), 6.0)
    
    out_offset = batch_idx * C_out * H * W + c_out * H * W + h * W + w
    tl.store(out_ptr + out_offset, clamped_val, mask=mask)

@torch.fx.wrap
def fused_conv_sigmoid_mul_clamp(in_0, in_1, in_2, in_3):
    B, C_out, H, W = in_2.shape
    in_C = in_3.shape[1]
    
    out = torch.empty_like(in_2)
    grid = (B, H, W)
    fused_conv_sigmoid_mul_clamp_kernel[grid](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        B=B,
        C_out=C_out,
        H=H,
        W=W,
        in_C=in_C,
        BLOCK_SIZE=256
    )
    
    return out

def replacement_func():
    return fused_conv_sigmoid_mul_clamp