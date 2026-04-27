import torch
import torch.fx.proxy
import triton
import triton.language as tl
import threading

# ── Monkey-patch torch.unbind for safe FX symbolic tracing ───────────────────
# Problem: torch.unbind(proxy, dim=2) crashes FX symbolic tracing because
# it calls proxy.size(2) to determine the output count, and range(SymInt) fails.
# Fix: when the flag is active (during pattern tracing), intercept the call
# and manually create the FX graph node, returning a single Proxy that
# __getitem__(0) / __getitem__(1) can create downstream getitem nodes from.
_PATCH_ACTIVE = threading.local()
_ORIG_UNBIND = torch.unbind

def _fx_safe_unbind(input_tensor, dim=0):
    if (getattr(_PATCH_ACTIVE, 'active', False)
            and isinstance(input_tensor, torch.fx.proxy.Proxy)):
        # Create call_function[original_torch.unbind] node directly,
        # bypassing the shape-dependent output-count determination.
        return input_tensor.tracer.create_proxy(
            'call_function',
            _ORIG_UNBIND,         # target = original torch.unbind
            (input_tensor,),      # positional args (tensor only)
            {'dim': dim},         # dim as keyword arg (matches model.py)
        )
    return _ORIG_UNBIND(input_tensor, dim)

torch.unbind = _fx_safe_unbind   # global patch (safe: delegates to _ORIG_UNBIND
                                  # when flag is inactive or input is not a Proxy)


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    _PATCH_ACTIVE.active = True
    try:
        tmp_1  = in_2 * in_1
        tmp_2  = tmp_1 + in_0
        unbind = torch.unbind(tmp_2, dim=2)   # now works: returns single Proxy
        tmp_4  = unbind[0]                    # getitem(unbind, 0)
        tmp_5  = unbind[1]                    # getitem(unbind, 1)
        tmp_6  = tmp_5.permute(0, 2, 1)
        return (tmp_6, tmp_4)
    finally:
        _PATCH_ACTIVE.active = False


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Grid: (N, K). Each (b,k) program reads in_2 ONCE and writes both
# out0[b,k,d] (=tmp_4) and out1[b,d,k] (=tmp_6, permuted) directly —
# no [N,K,2,D] intermediate tensor.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 128}, num_warps=16),
    ],
    key=['K', 'D'],
)
@triton.jit
def _full_fused_kernel(
    in0_ptr,   # [2, D]
    in1_ptr,   # [1, 1, 2, D]
    in2_ptr,   # [N, K, 1, D]
    out0_ptr,  # [N, K, D]   = tmp_4  (contiguous)
    out1_ptr,  # [N, D, K]   = tmp_6  (contiguous, permuted)
    K,
    D,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)

    in2_val = tl.load(in2_ptr + b * K * D + k * D + d)

    in1_v0 = tl.load(in1_ptr + d)
    in0_v0 = tl.load(in0_ptr + d)
    tl.store(out0_ptr + b * K * D + k * D + d, in2_val * in1_v0 + in0_v0)

    in1_v1 = tl.load(in1_ptr + D + d)
    in0_v1 = tl.load(in0_ptr + D + d)
    tl.store(out1_ptr + b * D * K + d * K + k, in2_val * in1_v1 + in0_v1)


# ── Wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_mul_add_unbind_permute(in_0, in_1, in_2):
    N = in_2.shape[0]
    K = in_2.shape[1]
    D = in_2.shape[3]

    out0 = torch.empty((N, K, D), dtype=in_2.dtype, device=in_2.device)
    out1 = torch.empty((N, D, K), dtype=in_2.dtype, device=in_2.device)

    _full_fused_kernel[(N, K)](
        in_0, in_1, in_2,
        out0, out1,
        K, D,
    )

    # model returns (tmp_6, tmp_4) = (out1, out0)
    return (out1, out0)


def replacement_func():
    return fused_mul_add_unbind_permute