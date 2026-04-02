import operator
import torch
import torch.fx.proxy as _fx_proxy
import triton
import triton.language as tl


# ── Monkeypatch FX Proxy so that "+=" emits operator.iadd nodes ─────────────
# Standard FX Proxy.__iadd__ falls back to __add__ → creates 'add' nodes,
# but Dynamo-traced models produce 'iadd' nodes.  This patch aligns them so
# the SubgraphMatcher can find a match.
def _proxy_iadd(self, other):
    return self.tracer.create_proxy(
        'call_function', operator.iadd, (self, other), {}
    )

if not getattr(getattr(_fx_proxy.Proxy, '__iadd__', None), '_patched_iadd', False):
    _proxy_iadd._patched_iadd = True
    _fx_proxy.Proxy.__iadd__ = _proxy_iadd


# ── Pattern: iadd (in-place add) followed by transpose(1, 2) ────────────────
def pattern(in_0, in_1):
    in_1 += in_0                    # → operator.iadd node (via monkeypatch)
    tmp_2 = in_1.transpose(1, 2)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Fused Triton kernel: add (broadcast) + transpose(1, 2) ──────────────────
# BLOCK_SIZE == C == 128:
#   • Each program handles one full output row (w = pid, c = 0..127)
#   • in0 reads : c = arange(0,128)      → coalesced ✓
#   • out writes: pid*128 .. pid*128+127  → coalesced ✓
#   • in1 reads : c*19 + pid             → stride-19 (5 KB tensor, L1-resident)
#   • No masking: 19 × 128 = 2432 exactly
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_add_transpose_kernel(
    in0_ptr,                  # [128, 1]
    in1_ptr,                  # [1, 128, 19]
    out_ptr,                  # [1, 19, 128]
    BLOCK_SIZE: tl.constexpr, # 128  (== C)
):
    W: tl.constexpr = 19

    pid     = tl.program_id(0)                              # 0 ≤ pid < 19  (= w)
    # c cycles 0..127 sequentially within each program.
    # offsets = pid*128 + c, so (offsets % 128) == c  — compute c directly.
    c       = tl.arange(0, BLOCK_SIZE)                     # [0..127]
    offsets = pid * BLOCK_SIZE + c                          # store positions

    in1_idx = c * W + pid                                   # in1[0, c, w=pid]

    v0 = tl.load(in0_ptr + c)        # coalesced reads  ✓
    v1 = tl.load(in1_ptr + in1_idx)  # stride-19, L1-resident

    tl.store(out_ptr + offsets, v1 + v0)  # coalesced writes ✓


@torch.fx.wrap
def fused_add_transpose(in_0, in_1):
    """
    Fused: out[0, w, c] = in_1[0, c, w] + in_0[c, 0]
    in_0 : [128, 1]
    in_1 : [1, 128, 19]
    Returns a single tensor [1, 19, 128].
    """
    out = torch.empty((1, 19, 128), dtype=in_1.dtype, device=in_1.device)

    # 19 programs × 128 elements = 2432 (exact fit, no masking)
    fused_add_transpose_kernel[(19,)](
        in_0, in_1, out,
        BLOCK_SIZE=128,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_add_transpose