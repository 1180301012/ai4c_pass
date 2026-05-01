import torch
import triton
import triton.language as tl
from torch import inf


# ---------------------------------------------------------------------------
# Pattern
#
# Match ONLY the two elementwise multiplications:
#   tmp_6 = tmp_5 * in_4
#   tmp_8 = tmp_6 * tmp_7
#
# All three placeholders (tmp_5, in_4, tmp_7) have ONLY the matched muls
# as consumers in the target graph, so NOT_CONTAINED is never triggered.
# No __eq__/masked_fill_ nodes involved → no OP/TARGET_MISMATCH for eq.
#
# tmp_5 = deg[row]^(-0.5),  in_4 = edge_weight,  tmp_7 = deg[col]^(-0.5)
# The fused kernel computes: out[e] = tmp_5[e] * in_4[e] * tmp_7[e]
# ---------------------------------------------------------------------------
def pattern(tmp_5, in_4, tmp_7):
    tmp_6 = tmp_5 * in_4
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(tmp_5, in_4, tmp_7):
    return (tmp_5, in_4, tmp_7)


# ---------------------------------------------------------------------------
# Fused Triton kernel: out[e] = a[e] * b[e] * c[e]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_mul3_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = tl.load(c_ptr + offs, mask=mask, other=0.0)
    out = (a.to(tl.float32) * b.to(tl.float32) * c.to(tl.float32)).to(a.dtype)
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def fused_mul3(tmp_5, in_4, tmp_7):
    n = tmp_5.shape[0]
    out = torch.empty(n, dtype=in_4.dtype, device=in_4.device)

    def grid(meta):
        return ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_mul3_kernel[grid](tmp_5, in_4, tmp_7, out, n)
    return out


def replacement_func():
    return fused_mul3