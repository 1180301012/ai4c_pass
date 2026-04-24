import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass A: Fuse sub(−1) → masked_fill_
#
# Strategy: take tmp_3 (eq result) as an INPUT placeholder, not as a
# computed internal node.  This avoids:
#   1. Dead code (eq would be dead if computed inside pattern but not returned)
#   2. TARGET_MISMATCH (eq vs __eq__ call_method name)
#
# Pattern covers: sub + masked_fill_  → 1 Triton kernel
# cumsum and eq remain in the graph (already executed by the time this kernel
# receives its inputs: cumsum result as in_1, and eq result as in_0).
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    # in_1 = cumsum result  [B, S]
    # in_0 = eq mask        [B, S]  (placeholder — matched externally)
    tmp_2 = in_1 - 1
    tmp_2 = tmp_2.masked_fill_(in_0, 1)
    tmp_5 = tmp_2.unsqueeze(0)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused subtract + masked_fill
# Grid: (B,) — one program per batch row.
# in_1 = cumsum result  [B, S]
# in_0 = eq mask        [B, S]
# out   = cumsum-1, then masked-fill (set 1 where eq==1)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64},  num_warps=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4),
        triton.Config({'BLOCK_S': 256}, num_warps=4),
        triton.Config({'BLOCK_S': 512}, num_warps=8),
        triton.Config({'BLOCK_S': 1024}, num_warps=8),
    ],
    key=['S'],
)
@triton.jit
def _fused_sub_masked_fill_kernel(
    in0_ptr, in1_ptr, out_ptr,
    S, in_stride_b, BLOCK_S: tl.constexpr,
):
    pid_b   = tl.program_id(0)
    base_in = in_stride_b * pid_b
    offs    = tl.arange(0, BLOCK_S)
    mask    = offs < S

    in0 = tl.load(in0_ptr + base_in + offs, mask=mask, other=0)
    in1 = tl.load(in1_ptr + base_in + offs, mask=mask, other=0)

    cs_sub1 = in1 - 1
    out     = tl.where(in0 == 0, 1, cs_sub1)

    tl.store(out_ptr + base_in + offs, out, mask=mask)


@torch.fx.wrap
def _fused_sub_masked_fill(in_0, in_1):
    # in_0 = eq mask [B, S],  in_1 = cumsum result [B, S]
    B   = in_1.shape[0]
    S   = in_1.shape[1]
    out = torch.empty((B, S), dtype=torch.int64, device=in_1.device)
    _fused_sub_masked_fill_kernel[(B,)](in_0, in_1, out, S, in_0.stride(0))
    # Return unsqueezed view to match pattern output [1, B, S]
    return out.unsqueeze(0)


def replacement_func():
    return _fused_sub_masked_fill