import torch
import triton
import triton.language as tl


# ── Pattern ────────────────────────────────────────────────────────────────────
def pattern(in_1):
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


# ── Triton kernel ──────────────────────────────────────────────────────────────
# Fused sum(dim=2, keepdim=True) + elementwise divide.
# Grid (16,): each program handles one 8-element dim-2 row.
@triton.jit
def _sum_div_kernel(in_ptr, out_ptr):
    pid  = tl.program_id(0)
    offs = tl.arange(0, 8)
    row  = pid * 8
    v    = tl.load(in_ptr + row + offs)
    tl.store(out_ptr + row + offs, v / tl.sum(v, axis=0))


# ── Wrapper ────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fuse_sum_div(in_1):
    out = torch.empty_like(in_1)
    _sum_div_kernel[(16,)](in_1, out)
    return out


# ── Replacement hook ───────────────────────────────────────────────────────────
def replacement_func():
    return fuse_sum_div