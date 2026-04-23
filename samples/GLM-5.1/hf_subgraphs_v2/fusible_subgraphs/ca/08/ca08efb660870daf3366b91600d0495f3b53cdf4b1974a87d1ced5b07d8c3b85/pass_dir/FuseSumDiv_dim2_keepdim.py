import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = x.sum(dim=2, keepdim=True)
    tmp_1 = x / tmp_0
    return tmp_1


def replacement_args(x):
    return (x,)


@triton.jit
def fused_norm_kernel(
    x_ptr, out_ptr,
    x_s1, x_s2, x_s3,
    out_s1, out_s2, out_s3,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
):
    pid = tl.program_id(0)

    # Each program processes one (b, c) slice
    # B=1 so we just use pid for channel index
    in_base = pid * x_s1
    out_base = pid * out_s1

    col_offsets = tl.arange(0, n_cols)

    # Sum across rows (dim=2)
    sum_vals = tl.zeros([n_cols], dtype=tl.float32)
    for r in range(n_rows):
        offsets = in_base + r * x_s2 + col_offsets * x_s3
        x_vals = tl.load(x_ptr + offsets).to(tl.float32)
        sum_vals += x_vals

    # Divide and write
    for r in range(n_rows):
        in_offsets = in_base + r * x_s2 + col_offsets * x_s3
        out_offsets = out_base + r * out_s2 + col_offsets * out_s3
        x_vals = tl.load(x_ptr + in_offsets).to(tl.float32)
        out_vals = x_vals / sum_vals
        tl.store(out_ptr + out_offsets, out_vals)


@torch.fx.wrap
def fused_sum_div(x):
    out = torch.empty_like(x)

    C = x.shape[1]
    n_rows = x.shape[2]
    n_cols = x.shape[3]

    fused_norm_kernel[(C,)](
        x_ptr=x,
        out_ptr=out,
        x_s1=x.stride(1),
        x_s2=x.stride(2),
        x_s3=x.stride(3),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        out_s3=out.stride(3),
        n_rows=n_rows,
        n_cols=n_cols,
    )

    return out


def replacement_func():
    return fused_sum_div