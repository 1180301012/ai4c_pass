import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 2048}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_D': 2048}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_D': 2048}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_D': 2048}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_D': 4096}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_D': 4096}, num_warps=16, num_stages=1),
    ],
    key=['D'],
)
@triton.jit
def _rms_norm_1e6_bf16_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # Load input (bfloat16) and upcast to float32
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute RMSNorm
    x2 = x * x
    mean_x2 = tl.sum(x2, axis=0) * (1.0 / D)
    rstd = tl.rsqrt(mean_x2 + 1e-6)
    x_hat = x * rstd

    # Load weight (bfloat16) and apply
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = (x_hat * w).to(tl.bfloat16)

    tl.store(out_ptr + row * D + cols, out, mask=mask)


@torch.fx.wrap
def fused_rms_norm_1e6_bf16(weight, x):
    orig_shape = x.shape
    D = orig_shape[-1]
    N_rows = x.numel() // D
    x_2d = x.view(N_rows, D)
    out = torch.empty_like(x_2d)
    _rms_norm_1e6_bf16_kernel[(N_rows,)](x_2d, weight, out, D)
    return out.view(orig_shape)


def pattern(in_0, in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2)


def replacement_func():
    return fused_rms_norm_1e6_bf16