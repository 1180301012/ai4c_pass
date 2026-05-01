import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_dispatch import _dispatch

@triton.jit
def _causal_mask_kernel_N10(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    NEG_INF = -3.4028234663852886e+38
    out_val = tl.where(cols > row,
                       tl.full([BLOCK_N], NEG_INF, dtype=tl.float32),
                       tl.zeros([BLOCK_N], dtype=tl.float32))
    tl.store(out_ptr + row * N + cols, out_val, mask=col_mask)

@torch.fx.wrap
def _build_causal_N10():
    out = torch.empty((1, 1, 10, 10), dtype=torch.float32, device='cuda:0')
    _causal_mask_kernel_N10[(10,)](out, N=10, BLOCK_N=16)
    return out

def pattern():
    tmp_1 = torch.arange(0, 10, device=device(type='cuda', index=0))
    tmp_2 = torch.full((10, 10), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(10, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    return (tmp_10,)

def replacement_args():
    return ()

def replacement_func():
    return _build_causal_N10