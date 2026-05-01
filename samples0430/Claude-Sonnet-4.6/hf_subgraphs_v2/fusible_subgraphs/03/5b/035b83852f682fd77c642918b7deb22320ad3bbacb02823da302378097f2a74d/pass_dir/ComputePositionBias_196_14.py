import torch
import triton
import triton.language as tl


# ── Precompute the constant position-bias tensor (CPU, float32) ──────────────
def _make_pos_bias_data():
    N = 14
    data = []
    for i in range(N * N):
        a, b = i // N, i % N
        for j in range(N * N):
            c, d = j // N, j % N
            dx = float(d - b)
            dy = float(c - a)
            data.extend([dx, dy, dx * dx + dy * dy])
    return data

_POS_BIAS_DATA = _make_pos_bias_data()   # list of 115248 floats
_POS_BIAS_TENSOR = None                  # lazy-init cache


# ── Triton layer-norm kernel (row-parallel, float32 accumulation) ────────────
@triton.jit
def layer_norm_fwd_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program per row (token). Processes N elements in parallel."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    base = row * N

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_sub = x - mean
    var = tl.sum(x_sub * x_sub, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    x_norm = x_sub * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = w * x_norm + b
    tl.store(out_ptr + base + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


# ── Fused replacement for normalized_shape=(432,) ────────────────────────────
_POS_BIAS_TENSOR_432 = None

@torch.fx.wrap
def fused_ln_posbias_432(in_0, in_1, in_2):
    """Replace layer_norm(432) + constant position-bias construction."""
    global _POS_BIAS_TENSOR_432
    if _POS_BIAS_TENSOR_432 is None:
        _POS_BIAS_TENSOR_432 = torch.as_tensor(
            _POS_BIAS_DATA, dtype=torch.float32
        ).reshape(1, 196, 196, 3)

    # Layer norm via Triton (in_2: cuda, in_0/in_1: may be cpu)
    dev = in_2.device
    w = in_1.to(dev)
    b = in_0.to(dev)
    x = in_2.contiguous()
    out_ln = torch.empty_like(x)
    M = x.shape[0] * x.shape[1]   # 1*196 = 196
    N = x.shape[2]                 # 432
    layer_norm_fwd_kernel[(M,)](
        x.reshape(M, N), w, b, out_ln.reshape(M, N),
        M, N, 1e-6,
        BLOCK_SIZE=512,
    )
    return (_POS_BIAS_TENSOR_432, out_ln)


# ── Pattern and wiring ────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (432,), in_1, in_0, 1e-06)
    tmp_3 = torch.zeros(1, 196, 196, 3)
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = tmp_15
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_9.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    return (tmp_3, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_ln_posbias_432