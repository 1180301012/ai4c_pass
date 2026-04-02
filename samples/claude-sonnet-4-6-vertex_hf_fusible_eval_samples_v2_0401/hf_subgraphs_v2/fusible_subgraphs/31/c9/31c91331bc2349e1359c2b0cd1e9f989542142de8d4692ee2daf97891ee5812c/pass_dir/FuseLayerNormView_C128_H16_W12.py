import torch
import triton
import triton.language as tl

# Pass 2 of 2: LayerNorm + view → produces tmp_12 (single output)
# C=128, H=16, W=12, N=192
# input (tmp_10): [1, 192, 128]
# weight (in_1): [128], bias (in_0): [128]
# output (tmp_12): [1, 16, 12, 128]

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['N_VAL', 'C_VAL'],
)
@triton.jit
def _layernorm_kernel_C128(
    x_ptr, w_ptr, b_ptr,
    out_ptr,
    eps,
    N_VAL: tl.constexpr,
    C_VAL: tl.constexpr,
):
    n = tl.program_id(0)
    c_range = tl.arange(0, C_VAL)

    # Load row n of x [1, N, C]
    x_raw = tl.load(x_ptr + n * C_VAL + c_range)
    x = x_raw.to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=0) / C_VAL
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / C_VAL
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + c_range).to(tl.float32)
    b = tl.load(b_ptr + c_range).to(tl.float32)
    out = (diff * rstd) * w + b

    # Store to output (stored as [1,N,C], same memory as [1,H,W,C])
    tl.store(out_ptr + n * C_VAL + c_range, out.to(x_raw.dtype))


# Pattern: layer_norm(tmp_10, (128,), in_1, in_0, 1e-06) → view → tmp_12 (SINGLE output)
def pattern(tmp_10, in_1, in_0):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (128,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 16, 12, 128)
    return tmp_12


def replacement_args(tmp_10, in_1, in_0):
    return (tmp_10, in_1, in_0)


@torch.fx.wrap
def _layernorm_view_C128_H16_W12(tmp_10, in_1, in_0):
    N_val = 192
    C_val = 128
    H_val = 16
    W_val = 12

    # Allocate output directly in [1, H, W, C] shape (same mem as [1, N, C])
    out = torch.empty(1, H_val, W_val, C_val, dtype=tmp_10.dtype, device=tmp_10.device)

    _layernorm_kernel_C128[(N_val,)](
        tmp_10, in_1, in_0,
        out,
        1e-6,
        N_VAL=N_val,
        C_VAL=C_val,
    )

    return out


def replacement_func():
    return _layernorm_view_C128_H16_W12