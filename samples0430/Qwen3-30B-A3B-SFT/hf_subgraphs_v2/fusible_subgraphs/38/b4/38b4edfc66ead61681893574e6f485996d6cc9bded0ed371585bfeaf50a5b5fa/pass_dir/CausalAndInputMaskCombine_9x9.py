import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.causal_mask_shared import causal_mask_dispatch


# ─── Pattern: pure causal mask generation (N=9) ───────────────────────────────
# Matches: full(-inf) → arange → +1 → view → lt → masked_fill_ → to(f32)
#          → __getitem__ → expand(1,1,9,9)

def pattern():
    tmp_1 = torch.full((9, 9), -3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.arange(9, device=device(type='cuda', index=0))
    tmp_3 = tmp_2 + 1
    tmp_4 = tmp_3.view(9, 1)
    tmp_5 = tmp_2 < tmp_4
    tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
    tmp_7 = tmp_1.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, 9, 9)
    return tmp_9


def replacement_args():
    return ("causal_9",)


def replacement_func():
    return causal_mask_dispatch