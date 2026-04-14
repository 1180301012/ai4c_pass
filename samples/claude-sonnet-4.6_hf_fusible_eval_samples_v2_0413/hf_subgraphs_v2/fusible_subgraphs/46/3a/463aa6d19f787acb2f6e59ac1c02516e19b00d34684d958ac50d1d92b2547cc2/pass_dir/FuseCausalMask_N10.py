import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.causal_mask_kernel import build_causal_mask_kernel


def pattern():
    tmp_1 = torch.arange(0, 10, device=device(type='cuda', index=0))
    tmp_2 = torch.full((10, 10), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(10, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    return tmp_7


def replacement_args():
    return ()


@torch.fx.wrap
def causal_mask_N10():
    N = 10
    out = torch.empty((N, N), dtype=torch.float32, device='cuda:0')
    build_causal_mask_kernel[(N,)](out, N=N, BLOCK_N=16)
    return out


def replacement_func():
    return causal_mask_N10