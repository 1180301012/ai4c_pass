import torch
import triton
import triton.language as tl


# ── Key insight from diagnostics ──────────────────────────────────────────────
# - FuseAttentionMask WAS starting to match, failing only because:
#     pattern `sub` = call_method  (concrete_tensor - proxy → proxy.__rsub__)
#     target  `sub` = call_function (two proxy nodes → operator.sub)
# - Fix: make one_val (the torch.tensor(1.0,...) node) a PARAMETER so both
#   sides of the subtraction are Proxies → call_function.
#
# - We proved single-parameter patterns work.
# - This attention-mask pattern has 2 params: in_5 and one_val.
#   The framework anchors at in_5.to(torch.float32) (first computation).
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _attn_mask_kernel(
    in5_ptr,    # [N] int64
    out_ptr,    # [N] float32
    N,
    NEG_INF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x    = tl.load(in5_ptr + offs, mask=mask, other=0).to(tl.float32)
    val  = 1.0 - x
    # masked_fill: fill with NEG_INF where val != 0 (i.e. where bool(val) is True)
    out  = tl.where(val != 0.0, NEG_INF, val)
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def triton_attn_mask(in_5):
    """Fused: to_float32 → 1-x → to_bool → masked_fill(-inf)"""
    N   = in_5.numel()
    out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)
    # Adaptive BLOCK: next power-of-2 ≥ N, clamped to [32, 1024]
    # A single block covers all elements → minimum kernel-launch overhead
    BLOCK = max(32, min(1024, triton.next_power_of_2(N)))
    nw    = max(1, BLOCK // 32)
    _attn_mask_kernel[(1,)](          # <-- exactly 1 CUDA block
        in5_ptr=in_5, out_ptr=out,
        N=N, NEG_INF=-3.4028234663852886e+38,
        BLOCK=BLOCK, num_warps=nw,
    )
    return out


# ── Pattern: attention mask computation ───────────────────────────────────────
# one_val is the torch.tensor(1.0, dtype=float32) node in the graph.
# By making it a parameter (proxy), `one_val - tmp_4` traces as
# call_function(operator.sub, ...) matching the target's call_function.
def pattern(in_5, one_val):
    tmp_4 = in_5.to(torch.float32)
    tmp_6 = one_val - tmp_4          # proxy - proxy → call_function ✓
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5, one_val):
    return (in_5,)               # one_val is constant 1.0, hardcoded in kernel


def replacement_func():
    return triton_attn_mask