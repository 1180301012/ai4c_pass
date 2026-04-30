import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches:
#   m = torch.max(x, -1, keepdim=True)[0]
#   y = torch.nn.functional.softmax(m.expand_as(x) - x, dim=-1)
# This is mathematically equivalent to softmax(-x, dim=-1).
def pattern(in_0):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _neg_softmax_lastdim_kernel(
    x_ptr,
    out_ptr,
    row_stride_x,
    row_stride_out,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row_id * row_stride_x + cols, mask=mask, other=0.0)
    vals = -x.to(tl.float32)
    vals = tl.where(mask, vals, -float("inf"))

    row_max = tl.max(vals, axis=0)
    vals = vals - row_max
    numer = tl.exp(vals)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_ptr + row_id * row_stride_out + cols, out, mask=mask)


@torch.fx.wrap
def triton_neg_softmax_lastdim(in_0):
    n_cols = in_0.shape[-1]
    x_2d = in_0.view(-1, n_cols)
    out = torch.empty_like(in_0)
    out_2d = out.view(-1, n_cols)
    n_rows = x_2d.shape[0]

    # The target graphs all use last-dimension 512, but keeping a dynamic
    # fallback block size selection here helps robustness.
    if n_cols <= 128:
        block_size = 128
        num_warps = 2
    elif n_cols <= 256:
        block_size = 256
        num_warps = 4
    elif n_cols <= 512:
        block_size = 512
        num_warps = 8
    elif n_cols <= 1024:
        block_size = 1024
        num_warps = 8
    else:
        # The pattern is only expected on small attention maps here.
        raise RuntimeError(f"Unsupported softmax width: {n_cols}")

    _neg_softmax_lastdim_kernel[(n_rows,)](
        x_2d,
        out_2d,
        x_2d.stride(0),
        out_2d.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_neg_softmax_lastdim