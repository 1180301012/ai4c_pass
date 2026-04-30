import torch
import triton
import triton.language as tl

@triton.jit
def normalize_divide_kernel(
    in_1_ptr, in_2_ptr, out_2_ptr, out_4_ptr,
    in_1_numel, in_2_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """First kernel: Normalize-divide for both in_1 and in_2."""
    pid = tl.program_id(0)
    
    # Process in_1 normalization (512 elements, shape [1, 512])
    if pid == 0:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < in_1_numel
        
        x = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        x_sq = x_fp32 * x_fp32
        sum_sq = tl.sum(x_sq, axis=0)
        rsqrt = tl.math.rsqrt(sum_sq + 1e-6)
        x_norm = x * rsqrt
        tl.store(out_2_ptr + offsets, x_norm, mask=mask)
    
    # Process in_2 normalization (512 elements, shape [1, 1, 512])
    elif pid == 1:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < in_2_numel
        
        x = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        x_sq = x_fp32 * x_fp32
        sum_sq = tl.sum(x_sq, axis=0)
        rsqrt = tl.math.rsqrt(sum_sq + 1e-6)
        x_norm = x * rsqrt
        tl.store(out_4_ptr + offsets, x_norm, mask=mask)

@triton.jit
def exp_mul_kernel(
    in_0_ptr, out_4_ptr, out_6_ptr,
    out_6_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Second kernel: exp(scalar) * tensor."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_6_numel
    
    tensor = tl.load(out_4_ptr + offsets, mask=mask, other=0.0)
    scalar = tl.load(in_0_ptr).to(tl.float32)
    exp_scalar = tl.exp(scalar)
    result = tensor * exp_scalar
    tl.store(out_6_ptr + offsets, result, mask=mask)

def pattern(in_0, in_1, in_2):
    tmp_1 = in_1.norm(p = 2, dim = -1, keepdim = True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p = 2, dim = -1, keepdim = True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return tmp_6, tmp_4, tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def fused_all_ops_wrapper(in_0, in_1, in_2):
    in_1_numel = in_1.numel()  # 512
    in_2_numel = in_2.numel()  # 512
    
    BLOCK_SIZE = 512
    
    out_2 = torch.empty_like(in_1)
    out_4 = torch.empty_like(in_2)
    out_6 = torch.empty_like(in_2)
    
    # First kernel: Normalize-divide for in_1 and in_2
    normalize_divide_kernel[(2,)](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_2_ptr=out_2,
        out_4_ptr=out_4,
        in_1_numel=in_1_numel,
        in_2_numel=in_2_numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Second kernel: exp(scalar) * tensor
    exp_mul_kernel[(1,)](
        in_0_ptr=in_0,
        out_4_ptr=out_4,
        out_6_ptr=out_6,
        out_6_numel=in_2_numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_6, out_4, out_2

def replacement_func():
    return fused_all_ops_wrapper