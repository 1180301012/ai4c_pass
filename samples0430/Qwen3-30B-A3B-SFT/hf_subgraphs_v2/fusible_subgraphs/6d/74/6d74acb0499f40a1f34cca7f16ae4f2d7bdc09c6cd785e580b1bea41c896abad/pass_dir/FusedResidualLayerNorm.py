import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['N', 'n_rows'],
)
@triton.jit
def _fused_residual_layernorm_kernel(
    bias_ptr, weight_ptr, x_ptr, y_ptr, out_ptr,
    N, n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Preload weight and bias so the memory subsystem can fetch them early
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    sum_z  = tl.sum(z,     axis=0)
    sum_z2 = tl.sum(z * z, axis=0)

    mean = sum_z  / N
    var  = sum_z2 / N - mean * mean

    z_c     = tl.where(mask, z - mean, 0.0)
    inv_std = 1.0 / tl.sqrt(var + 1e-7)
    z_norm  = z_c * inv_std

    out = z_norm * w + b

    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def fused_residual_layernorm(in_0, in_1, in_2, in_3):
    # in_0: bias  [N]
    # in_1: weight [N]
    # in_2: residual [*, N]
    # in_3: input [*, N]
    N = in_2.shape[-1]
    n_rows = in_2.numel() // N

    out = torch.empty(in_2.shape, dtype=torch.float32, device=in_2.device)

    _fused_residual_layernorm_kernel[(n_rows,)](
        in_0, in_1, in_2, in_3, out,
        N, n_rows,
    )

    return out


def replacement_func():
    return fused_residual_layernorm