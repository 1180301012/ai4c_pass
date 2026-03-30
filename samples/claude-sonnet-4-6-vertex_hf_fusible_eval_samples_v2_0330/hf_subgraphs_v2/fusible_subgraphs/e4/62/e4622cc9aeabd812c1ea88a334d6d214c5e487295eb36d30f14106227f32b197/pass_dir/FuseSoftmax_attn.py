"""
Pass: FuseSoftmax_attn

Pattern: torch.nn.functional.softmax(x, dim=-1)
Replace: Triton row-wise softmax kernel
         Faster than PyTorch's generic softmax for the attention weight shape
         [bsz*num_heads, seq_len, seq_len] = [8, 150, 150] → 1200 rows × 150 cols
"""
import torch
import triton
import triton.language as tl


# ─── Triton online softmax (Milakov & Gimelshein style) ───────────────────────
# Grid: (n_rows,)  – one program per row of the last dimension
# Works for any 2-D layout after reshaping to (n_rows, n_cols).

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=8),
        triton.Config({'BLOCK': 128}, num_warps=4),
    ],
    key=['n_cols'],
)
@triton.jit
def _softmax_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    # Load row (with -inf padding for out-of-bounds)
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # Numerically-stable softmax
    x_max  = tl.max(x, axis=0)
    x_exp  = tl.exp(x - x_max)
    x_sum  = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum

    tl.store(out_ptr + row * stride_row + offs, result, mask=mask)


@torch.fx.wrap
def triton_softmax_last_dim(x):
    """
    Computes softmax over the last dimension of x (any shape).
    Equivalent to torch.nn.functional.softmax(x, dim=-1).
    """
    orig_shape = x.shape
    n_cols     = orig_shape[-1]
    n_rows     = x.numel() // n_cols

    # Flatten to 2-D for the kernel
    x_2d  = x.reshape(n_rows, n_cols).contiguous()
    out   = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)

    _softmax_kernel[(n_rows,)](
        x_2d, out,
        n_rows, n_cols,
        stride_row=n_cols,  # contiguous after reshape
    )
    return out.reshape(orig_shape)


# ─── Pattern ──────────────────────────────────────────────────────────────────

def pattern(x):
    return torch.nn.functional.softmax(x, dim=-1)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_softmax_last_dim