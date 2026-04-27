import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Probe: try matching just torch.cat to discover if the compiled graph
# uses torch.cat or aten.cat.default as the node target.
# ---------------------------------------------------------------------------

def pattern(x):
    out = torch.cat([x], 1)
    return out


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: row-wise L2 normalisation for a 2-D tensor [N, D]
#   - Each programme handles exactly one row.
#   - The squared sum is accumulated in float32 for numerical stability
#     regardless of the input dtype (fp32 / fp16 / bf16).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 1024}, num_warps=4),
        triton.Config({'BLOCK_D': 1024}, num_warps=8),
        triton.Config({'BLOCK_D': 512},  num_warps=4),
        triton.Config({'BLOCK_D': 512},  num_warps=8),
        triton.Config({'BLOCK_D': 256},  num_warps=4),
    ],
    key=['D'],
)
@triton.jit
def _l2_normalize_kernel(
    x_ptr,
    out_ptr,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_base = row_idx * D

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    # Load one row of the input (masked elements → 0.0)
    x = tl.load(x_ptr + row_base + offsets, mask=mask, other=0.0)

    # Upcast to float32 for the norm computation
    x_f32 = x.to(tl.float32)

    # Compute squared L2 norm of the row
    sq_sum = tl.sum(x_f32 * x_f32, axis=0)
    norm = tl.sqrt(sq_sum)

    # Guard against divide-by-zero (matches PyTorch default eps=1e-12)
    safe_norm = tl.where(norm > 1e-12, norm, 1e-12)

    # Normalise (division done in fp32, then cast back)
    out_f32 = x_f32 / safe_norm
    out = out_f32.to(x.dtype)

    tl.store(out_ptr + row_base + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX does not trace into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_l2_normalize(in_0):
    N, D = in_0.shape
    out = torch.empty_like(in_0)

    _l2_normalize_kernel[(N,)](
        x_ptr=in_0,
        out_ptr=out,
        N=N,
        D=D,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument, returns the callable (not a call)
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_l2_normalize