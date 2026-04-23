import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.14433756729740643
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7

def replacement_args(in_1, in_0):
    scale_val = 0.14433756729740643
    min_val = 1e-05
    return (in_1, in_0, scale_val, min_val)

@triton.jit
def row_normalize_kernel(
    x_ptr,  # [N, M]
    in0_ptr,  # scalar [1]
    scale_val,
    min_val,
    out_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr = 256
):
    row_idx = tl.program_id(0)
    start_idx = row_idx * M
    
    in0_val = tl.load(in0_ptr)
    
    sh = tl.zeros([1], dtype=tl.float32)
    
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (start_idx + M)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x
    tl.atomic_add(sh, tl.sum(x_sq))
    tl.sync()
    
    sum_sq = sh[0]
    norm = tl.sqrt(sum_sq) * scale_val
    norm = tl.max(norm, min_val)
    
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (start_idx + M)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / norm * in0_val
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def normalize_and_divide(x, in0, scale_val, min_val):
    N = x.shape[0]
    M = x.shape[1]
    out = torch.empty_like(x)
    
    grid = (N,)
    row_normalize_kernel[grid](
        x_ptr=x,
        in0_ptr=in0,
        scale_val=scale_val,
        min_val=min_val,
        out_ptr=out,
        N=N,
        M=M,
        BLOCK_SIZE=256
    )
    return out

def replacement_func():
    return normalize_and_divide