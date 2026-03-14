import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    b = pid // (H * W)
    rem = pid % (H * W)
    h = rem // W
    w = rem % W
    
    base_offset = b * stride_b + h * stride_h + w * stride_w
    
    sum_val = 0.0
    
    for ch_offset in range(0, C, BLOCK_SIZE):
        ch_offsets = ch_offset + tl.arange(0, BLOCK_SIZE)
        mask = ch_offsets < C
        
        offsets = base_offset + ch_offsets * stride_c
        
        in_0_vals = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
        in_1_vals = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
        
        prod = in_0_vals * in_1_vals
        sum_val += tl.sum(prod, axis=0)
    
    output = tl.sigmoid(sum_val)
    
    out_offset = b * (H * W) + h * W + w
    tl.store(out_ptr + out_offset, output)


def fused_mul_sum_sigmoid(in_0, in_1):
    B, C, H, W = in_0.shape
    assert in_1.shape == (B, C, H, W), "Input shapes must match"
    
    out = torch.empty((B, H, W), device=in_0.device, dtype=in_0.dtype)
    
    num_outputs = B * H * W
    grid = (num_outputs,)
    
    fused_mul_sum_sigmoid_kernel[grid](
        in_0, in_1, out,
        B, C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
    )
    
    out = out.unsqueeze(1)
    return out


@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    return fused_mul_sum_sigmoid(in_0, in_1)


def replacement_func():
    return kernel_wrapper