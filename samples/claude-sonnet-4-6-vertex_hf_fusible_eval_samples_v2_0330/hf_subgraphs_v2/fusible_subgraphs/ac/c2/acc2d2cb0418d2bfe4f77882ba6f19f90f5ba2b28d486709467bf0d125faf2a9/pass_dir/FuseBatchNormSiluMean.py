import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    z = torch.nn.functional.silu(y, inplace=True)
    m = z.mean((2, 3), keepdim=True)
    return z, m


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=4),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8192}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _bn_silu_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    # One program per (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Load batch-norm parameters for this channel (compute in fp32)
    mean_c  = tl.load(running_mean_ptr + c).to(tl.float32)
    var_c   = tl.load(running_var_ptr  + c).to(tl.float32)
    w_c     = tl.load(weight_ptr       + c).to(tl.float32)
    b_c     = tl.load(bias_ptr         + c).to(tl.float32)

    inv_std   = 1.0 / tl.sqrt(var_c + 1e-5)
    scale     = inv_std * w_c
    bias_term = b_c - mean_c * scale

    base = (n * C + c) * HW
    acc  = 0.0

    n_iters = (HW + BLOCK_HW - 1) // BLOCK_HW
    for i in range(n_iters):
        offsets = i * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask    = offsets < HW

        # Load input, upcast to fp32 for accuracy
        x_val = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

        # BatchNorm (inference)
        y_val = x_val * scale + bias_term

        # SiLU: y * sigmoid(y)
        z_val = y_val * tl.sigmoid(y_val)

        # Store to output (Triton auto-casts fp32 → pointer dtype)
        tl.store(out_ptr + base + offsets, z_val, mask=mask)

        # Accumulate for spatial mean
        acc += tl.sum(tl.where(mask, z_val, 0.0))

    # Write spatial mean for (n, c): layout [N, C] → index n*C + c
    tl.store(mean_out_ptr + n * C + c, acc / HW)


@torch.fx.wrap
def fused_bn_silu_mean(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    HW = H * W

    out      = torch.empty_like(x)
    # mean_out shape [N, C, 1, 1]; written through a [N*C] flat view
    mean_out = torch.empty(N, C, 1, 1, dtype=x.dtype, device=x.device)
    mean_flat = mean_out.view(N * C)

    grid = (N * C,)

    _bn_silu_mean_kernel[grid](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        mean_flat,
        C,
        HW,
    )

    return out, mean_out


def replacement_func():
    return fused_bn_silu_mean