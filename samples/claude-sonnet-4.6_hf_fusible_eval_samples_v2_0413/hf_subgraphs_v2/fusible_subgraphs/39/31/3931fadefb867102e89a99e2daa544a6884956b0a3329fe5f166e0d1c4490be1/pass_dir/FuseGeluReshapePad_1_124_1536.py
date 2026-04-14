import torch
import inspect
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: GELU -> reshape -> reshape -> pad   (high-level ops)
# Input:  [1, 124, 1536]   Output: [1, 249, 768]
# ---------------------------------------------------------------------------
# KEY FINDING: The compiled model uses torch._C._nn.pad (not F.pad).
# Dynamo traces through F.pad's Python body and records the underlying
# torch._C._nn.pad call.  F.gelu stays as F.gelu (1 arg, no default added
# since it's a C builtin with no inspectable signature, so ForceArgsTracer
# falls back to the original 1-arg call).
# The two reshapes stay as call_method nodes (ForceArgsTracer leaves them).
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_0 = torch.nn.functional.gelu(x)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch._C._nn.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Fused Triton kernel: GELU (erf-based) + implicit reshape + zero-pad
# Shapes are fixed: input [1,124,1536], output [1,249,768].
# Using constexpr sizes so the compiler can eliminate dead branches.
# num_warps=8 for better SFU and memory-latency hiding on Ampere.
# ---------------------------------------------------------------------------

_N_GELU  = 190464   # 1 * 124 * 1536
_N_OUT   = 191232   # 1 * 249 * 768
_BLOCK   = 1024
_NBLOCKS = (_N_OUT + _BLOCK - 1) // _BLOCK   # 187

@triton.jit
def fused_gelu_pad_kernel(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    N_GELU:     tl.constexpr,
    N_OUT:      tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N_OUT
    in_mask = mask & (offsets < N_GELU)

    x     = tl.load(x_ptr + offsets, mask=in_mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # GELU (erf-based): x * 0.5 * (1 + erf(x/sqrt(2)))
    # For the pad region x=0.0, so GELU(0)=0 automatically.
    INV_SQRT2 = 0.7071067811865476
    gelu = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * INV_SQRT2))

    tl.store(out_ptr + offsets, gelu.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_gelu_reshape_pad(x):
    # x: [1, 124, 1536]  →  out: [1, 249, 768]
    out = torch.empty((1, 249, 768), dtype=x.dtype, device=x.device)
    fused_gelu_pad_kernel[(_NBLOCKS,)](
        x, out,
        BLOCK_SIZE=_BLOCK,
        N_GELU=_N_GELU,
        N_OUT=_N_OUT,
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_gelu_reshape_pad