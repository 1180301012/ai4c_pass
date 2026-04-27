import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_kernel import fuse_attn_mask_kernel_flat


# ── pattern ──────────────────────────────────────────────────────────────────

def pattern(tmp_9_ext, in_0, tmp_13_ext):
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 13, 13)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = tmp_13_ext - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9_ext.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


# ── replacement_args ──────────────────────────────────────────────────────────

def replacement_args(tmp_9_ext, in_0, tmp_13_ext):
    return (in_0, "n13")


# ── shared dispatch wrapper (identical across all pass files) ─────────────────

def _fuse_n9(in_0):
    out = torch.empty((1, 1, 9, 9), dtype=torch.float32, device=in_0.device)
    fuse_attn_mask_kernel_flat[(1,)](in_0, out, N=9, N2=81, BLOCK=128)
    return out

def _fuse_n13(in_0):
    out = torch.empty((1, 1, 13, 13), dtype=torch.float32, device=in_0.device)
    fuse_attn_mask_kernel_flat[(1,)](in_0, out, N=13, N2=169, BLOCK=256)
    return out

@torch.fx.wrap
def fuse_mask_dispatch(in_0, route):
    if route == "n9":
        return _fuse_n9(in_0)
    elif route == "n13":
        return _fuse_n13(in_0)
    return _fuse_n13(in_0)


# ── replacement_func ──────────────────────────────────────────────────────────

def replacement_func():
    return fuse_mask_dispatch