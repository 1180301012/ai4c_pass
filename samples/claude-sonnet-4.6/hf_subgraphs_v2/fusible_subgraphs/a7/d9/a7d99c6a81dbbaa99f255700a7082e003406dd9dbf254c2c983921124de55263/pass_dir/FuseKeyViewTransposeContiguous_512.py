import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: in_4.view(1,1,-1,64).transpose(1,2).contiguous()  (key-states)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_4):
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    return tmp_9


def replacement_args(in_4):
    return (in_4,)


# ─────────────────────────────────────────────────────────────────────────────
# Triton copy kernel: read N=512 elements from src, write to dst
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def copy512_kernel(src_ptr, dst_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid    = tl.program_id(0)
    offs   = pid * BLOCK + tl.arange(0, BLOCK)
    mask   = offs < N
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@torch.fx.wrap
def key_view_transpose_contiguous(in_4):
    # in_4: [1,1,512] → out: [1,8,1,64]  (same 512 elements, just re-shaped)
    out = torch.empty((1, 8, 1, 64), dtype=in_4.dtype, device=in_4.device)
    copy512_kernel[(4,)](in_4, out, N=512, BLOCK=128, num_warps=4)
    return out


def replacement_func():
    return key_view_transpose_contiguous