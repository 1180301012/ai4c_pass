import torch
import triton
import triton.language as tl


@triton.jit
def conv_view_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    IN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
):
    # Single program: matmul [128,16] x [16] + bias + sigmoid
    out_ids = tl.arange(0, OUT_FEATURES)
    in_ids = tl.arange(0, IN_FEATURES)

    w = tl.load(weight_ptr + out_ids[:, None] * IN_FEATURES + in_ids[None, :]).to(tl.float32)
    inp = tl.load(input_ptr + in_ids).to(tl.float32)
    bias_val = tl.load(bias_ptr + out_ids).to(tl.float32)

    acc = tl.sum(w * inp[None, :], axis=1) + bias_val
    tl.store(out_ptr + out_ids, tl.sigmoid(acc))


@triton.jit
def normalize_dim3_kernel(
    x_ptr, out_ptr,
    TOTAL: tl.constexpr,
    COLS: tl.constexpr,
):
    # Single program: normalize 16 rows of 8 elements each
    rows = tl.arange(0, TOTAL // COLS)
    cols = tl.arange(0, COLS)
    offs = rows[:, None] * COLS + cols[None, :]

    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.sum(x, axis=1)
    tl.store(out_ptr + offs, x / s[:, None])


@torch.fx.wrap
def dispatch(a, b, c, route):
    if route == "conv":
        out = torch.empty(1, 2, 8, 8, dtype=a.dtype, device=a.device)
        conv_view_sigmoid_kernel[(1,)](
            input_ptr=a, weight_ptr=b, bias_ptr=c, out_ptr=out,
            IN_FEATURES=16, OUT_FEATURES=128,
            num_warps=4,
        )
        return out
    else:
        out = torch.empty(1, 2, 8, 8, dtype=a.dtype, device=a.device)
        normalize_dim3_kernel[(1,)](
            x_ptr=a, out_ptr=out,
            TOTAL=128, COLS=8,
            num_warps=1,
        )
        return out