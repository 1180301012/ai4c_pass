import torch
import triton
import triton.language as tl

# Pass 2 of 2: LayerNorm + view → produces tmp_12 (single output)
# C=256, H=8, W=6, N=48
# input (tmp_10): [1, 48, 256]
# weight (in_1): [256], bias (in_0): [256]
# output (tmp_12): [1, 8, 6, 256]

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
def _layernorm_kernel_C256(
    x_ptr, w_ptr, b_ptr,
    out_ptr,
    eps,
    N_VAL: tl.constexpr,
    C_VAL: tl.constexpr,
):
    n = tl.program_id(0)
    c_range = tl.arange(0, C_VAL)

    x_raw = tl.load(x_ptr + n * C_VAL + c_range)
    x = x_raw.to(tl.float32)

    mean = tl.sum(x, axis=0) / C_VAL
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / C_VAL
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + c_range).to(tl.float32)
    b = tl.load(b_ptr + c_range).to(tl.float32)
    out = (diff * rstd) * w + b

    tl.store(out_ptr + n * C_VAL + c_range, out.to(x_raw.dtype))


def pattern(tmp_10, in_1, in_0):
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 8, 6, 256)
    return tmp_12


def replacement_args(tmp_10, in_1, in_0):
    return (tmp_10, in_1, in_0)


@torch.fx.wrap
def _layernorm_view_C256_H8_W6(tmp_10, in_1, in_0):
    N_val = 48
    C_val = 256
    H_val = 8
    W_val = 6

    out = torch.empty(1, H_val, W_val, C_val, dtype=tmp_10.dtype, device=tmp_10.device)

    _layernorm_kernel_C256[(N_val,)](
        tmp_10, in_1, in_0,
        out,
        1e-6,
        N_VAL=N_val,
        C_VAL=C_val,
    )

    return out


def replacement_func():
    return _layernorm_view_C256_H8_W6