import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_3

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_relu_mean_kernel(
    in_ptr,
    out_ptr,
    B: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    stride_in_b,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    stride_out_b,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr
):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    in_base = in_ptr + b_idx * stride_in_b + c_idx * stride_in_c
    out_base = out_ptr + b_idx * stride_out_b + c_idx * stride_out_c
    sum_val = tl.zeros((1,), dtype=tl.float32)
    
    for h in range(H):
        for w in range(W):
            idx = in_base + h * stride_in_h + w * stride_in_w
            x = tl.load(idx)
            relu_val = tl.maximum(x, 0.0)
            tl.store(idx, relu_val)
            sum_val += relu_val
    
    mean_val = sum_val / (H * W)
    tl.store(out_base, mean_val)

@torch.fx.wrap
def fused_relu_mean(in_1):
    B, C, H, W = in_1.shape
    out = torch.empty((B, C, 1, 1), dtype=in_1.dtype, device=in_1.device)
    
    stride_in_b, stride_in_c, stride_in_h, stride_in_w = in_1.stride()
    stride_out_b, stride_out_c, stride_out_h, stride_out_w = out.stride()
    
    grid = (B, C)
    fused_relu_mean_kernel[grid](
        in_1,
        out,
        B, C, H, W,
        stride_in_b, stride_in_c, stride_in_h, stride_in_w,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_SIZE=128
    )
    
    return in_1, out

def replacement_func():
    return fused_relu_mean