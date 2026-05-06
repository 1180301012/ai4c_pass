import torch
import triton
import triton.language as tl

def pattern(in_0):
    n = in_0.shape[1]
    mask = torch.full((n, n), -3.4028234663852886e+38, device='cuda:0')
    idx = torch.arange(n, device='cuda:0')
    idx_one = idx + 1
    idx_col = idx.reshape(n, 1)
    mask_bool = idx < idx_col
    mask = mask.masked_fill(mask_bool, 0)
    mask = mask.to(torch.float32)
    mask_expanded = mask.expand(1, 1, n, n)
    in_0_expanded = in_0.expand(1, 1, n, n)
    in_0_expanded = in_0_expanded.to(torch.float32)
    one = torch.tensor(1.0, dtype=torch.float32)
    diff = one - in_0_expanded
    diff_bool = diff.to(torch.bool)
    mask_ret = mask_expanded.masked_fill(diff_bool, -3.4028234663852886e+38)
    mask_ret = mask_ret.bool()
    return mask_ret

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def triangular_mask_kernel(n_val, out_ptr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_val
    for i in offsets:
        for j in tl.arange(0, BLOCK_SIZE):
            if i < j:
                out_val = 0.0
            else:
                out_val = -3.4028234663852886e+38
            tl.store(out_ptr + (i * n_val) + j, out_val)

@torch.fx.wrap
def kernel_wrapper(in_0):
    n = in_0.shape[1]
    out = torch.empty((1, 1, n, n), device='cuda:0', dtype=torch.float32)
    triangular_mask_kernel[(1, 1)](n_val=n, out_ptr=out, BLOCK_SIZE=128)
    return out.to(torch.bool)

def replacement_func():
    return kernel_wrapper