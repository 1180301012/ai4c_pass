import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_forward_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_row_ptr = x_ptr + row * stride_x_row + cols
    x = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd
    y = y * w + b

    y_row_ptr = y_ptr + row * stride_y_row + cols
    tl.store(y_row_ptr, y, mask=mask)


@torch.fx.wrap
def shared_dispatch(b, w, x, seq_len, hidden):
    out = torch.empty((1, seq_len, hidden), device=x.device, dtype=x.dtype)
    if hidden == 32:
        layer_norm_forward_kernel[(seq_len,)](
            x,
            w,
            b,
            out,
            hidden,
            hidden,
            hidden,
            1e-12,
            BLOCK_SIZE=32,
            num_warps=1,
        )
        return out
    if hidden == 384:
        layer_norm_forward_kernel[(seq_len,)](
            x,
            w,
            b,
            out,
            hidden,
            hidden,
            hidden,
            1e-12,
            BLOCK_SIZE=512,
            num_warps=4,
        )
        return out
    layer_norm_forward_kernel[(seq_len,)](
        x,
        w,
        b,
        out,
        hidden,
        hidden,
        hidden,
        1e-12,
        BLOCK_SIZE=1024,
        num_warps=8,
    )
    return out