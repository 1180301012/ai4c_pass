import torch
import triton
import triton.language as tl


# Precompute position-bias data once at module import time (pure Python)
def _make_pos_bias_data():
    N = 14
    out = list()
    data = []
    data = []
    for i in range(N * N):
        a, b = i // N, i % N
        for j in range(N * N):
            c, d = j // N, j % N
            dx = float(d - b)
            dy = float(c - a)
            out.append(dx)
            out.append(dy)
            out.append(dx * dx + dy * dy)
    return out

_POS_BIAS_DATA = _make_pos_bias_data()   # 115248 floats
_POS_BIAS_TENSOR_432 = None              # cached CPU tensor


# Triton layer-norm kernel: one program per row, float32 accumulation
@triton.jit
def layer_norm_fwd_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
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


@torch.fx.wrap
def fused_ln_posbias_432(in_0, in_1, in_2):
    dev = in_2.device
    w = torch.as_tensor(in_1, device=dev)
    b = torch.as_tensor(in_0, device=dev)
    out_ln = torch.empty_like(in_2)
    M = in_2.shape[0] * in_2.shape[1]   # 196
    N = in_2.shape[2]                    # 432
    layer_norm_fwd_kernel[(M,)](
        in_2, w, b, out_ln,
        M, N, 1e-6,
        BLOCK_SIZE=512,
    )
    return out_ln


from pass_dir.shared_ln_kernel import ln_triton_dispatch  # shared dispatch


def pattern(in_0, in_1, in_2):
    return torch.nn.functional.layer_norm(in_2, (432,), in_1, in_0, 1e-06)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_432")


def replacement_func():
    return ln_triton_dispatch