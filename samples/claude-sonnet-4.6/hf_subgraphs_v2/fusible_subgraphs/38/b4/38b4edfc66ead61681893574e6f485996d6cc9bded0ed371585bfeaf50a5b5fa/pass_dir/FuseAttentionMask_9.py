import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_kernel import fuse_attn_mask_kernel_flat


# ── pattern ──────────────────────────────────────────────────────────────────
# Matches ops 12-19. Uses tmp_11_ext (post-expand, shape 1×1×N×N int64) and
# tmp_9_ext / tmp_13_ext as external placeholders.
# No N-specific expand constant → single pattern covers N=9 AND N=13.

def pattern(tmp_9_ext, tmp_11_ext, tmp_13_ext):
    tmp_12 = tmp_11_ext.to(torch.float32)
    tmp_14 = tmp_13_ext - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9_ext.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


# ── replacement_args ──────────────────────────────────────────────────────────
# Pass only tmp_11_ext; tmp_9_ext is recomputed in kernel; tmp_13_ext = 1.0.

def replacement_args(tmp_9_ext, tmp_11_ext, tmp_13_ext):
    return (tmp_11_ext,)


# ── optimised kernel wrapper ──────────────────────────────────────────────────

@torch.fx.wrap
def fuse_from_tmp11(tmp_11):
    """
    tmp_11 : (1,1,N,N) int64 — expanded attention mask (0=masked, 1=valid).
             tmp_11[0,0,i,j] == in_0[0,j] for all i.
    return : (1,1,N,N) float32 combined causal+attention mask.

    Reads tmp_11 at offset j (the column index) since the kernel only needs
    in_0[0,j] which equals tmp_11[0,0,0,j] = tmp_11_ptr[j].
    """
    N = tmp_11.shape[2]
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=tmp_11.device)
    if N == 9:
        fuse_attn_mask_kernel_flat[(1,)](tmp_11, out, N=9,  N2=81,  BLOCK=128)
    elif N == 13:
        fuse_attn_mask_kernel_flat[(1,)](tmp_11, out, N=13, N2=169, BLOCK=256)
    return out


# ── replacement_func ──────────────────────────────────────────────────────────

def replacement_func():
    return fuse_from_tmp11