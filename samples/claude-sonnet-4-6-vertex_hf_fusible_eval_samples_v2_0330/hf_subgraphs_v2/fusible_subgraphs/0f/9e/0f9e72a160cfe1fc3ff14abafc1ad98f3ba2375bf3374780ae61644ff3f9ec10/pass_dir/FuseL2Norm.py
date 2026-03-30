import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: L2-normalise a tensor along its last dimension.
# Matches both:
#   tmp_2 = in_1 / in_1.norm(p=2, dim=-1, keepdim=True)
#   tmp_4 = in_2 / in_2.norm(p=2, dim=-1, keepdim=True)
# Each produces a single observable output, so the single-output pattern works.
# ---------------------------------------------------------------------------
def pattern(x):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    result = x / norm
    return result


# ---------------------------------------------------------------------------
# Triton kernel: one program per row, full row loaded in a single BLOCK.
# Computes L2 norm in float32 for precision, stores back in the original dtype.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["n_cols"],
)
@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = row * n_cols
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x_raw = tl.load(x_ptr + start + offs, mask=mask, other=0.0)
    x = x_raw.to(tl.float32)
    norm_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(norm_sq)
    xn = x / norm

    tl.store(out_ptr + start + offs, xn.to(x_raw.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX treats it as a single leaf node).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_l2_normalize(x):
    """
    L2-normalise x along its last dimension.
    Works for any shape: reshapes to (n_rows, n_cols) internally.
    """
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    x_2d = x.contiguous().reshape(-1, n_cols)
    n_rows = x_2d.shape[0]

    out_2d = torch.empty_like(x_2d)

    l2_norm_kernel[(n_rows,)](
        x_2d,
        out_2d,
        n_rows,
        n_cols,
    )

    return out_2d.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Required interface
# ---------------------------------------------------------------------------
def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_l2_normalize