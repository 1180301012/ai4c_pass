import torch
import triton
import triton.language as tl


@triton.jit
def cat_kernel(x1_ptr, x2_ptr, out_ptr, N, C1, C2, HW, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    total_C = C1 + C2
    n_idx = pid // total_C
    c_idx = pid % total_C
    
    if n_idx >= N:
        return
    
    is_from_x1 = c_idx < C1
    
    for hw_start in range(0, HW, BLOCK_SIZE):
        offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        
        x1_idx = n_idx * C1 * HW + c_idx * HW + offs
        x2_idx = n_idx * C2 * HW + (c_idx - C1) * HW + offs
        out_idx = n_idx * total_C * HW + c_idx * HW + offs
        
        x1_val = tl.load(x1_ptr + x1_idx, mask=mask & is_from_x1, other=0.0)
        x2_val = tl.load(x2_ptr + x2_idx, mask=mask & (~is_from_x1), other=0.0)
        
        out_val = tl.where(is_from_x1, x1_val, x2_val)
        tl.store(out_ptr + out_idx, out_val, mask=mask)


@torch.fx.wrap
def triton_cat(tensors, dim):
    x1, x2 = tensors
    N, C1, H, W = x1.shape
    _, C2, _, _ = x2.shape
    HW = H * W
    total_C = C1 + C2
    
    out = torch.empty((N, total_C, H, W), dtype=x1.dtype, device=x1.device)
    
    BLOCK_SIZE = 1024
    grid = (N * total_C,)
    
    cat_kernel[grid](
        x1.contiguous(), x2.contiguous(), out,
        N, C1, C2, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(x1, x2):
    return torch.cat([x1, x2], 1)


def replacement_args(x1, x2):
    return ([x1, x2], 1)


def replacement_func():
    return triton_cat