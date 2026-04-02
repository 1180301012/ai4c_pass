import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    bn = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu = torch.nn.functional.silu(bn, inplace=True)
    mean = silu.mean((2, 3), keepdim=True)
    return silu, mean


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def bn_silu_mean_kernel(
    x_ptr, rm_ptr, rv_ptr, w_ptr, b_ptr,
    out_ptr, mean_ptr,
    C, HW, eps,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C

    # Load BN parameters in fp32 for accuracy
    mean_c = tl.load(rm_ptr + c).to(tl.float32)
    var_c = tl.load(rv_ptr + c).to(tl.float32)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)

    # Precompute scale and shift: y = (x - mean) / sqrt(var + eps) * w + b
    #                              y = x * (w / sqrt(var+eps)) + (b - mean * w / sqrt(var+eps))
    inv_std = 1.0 / tl.sqrt(var_c + eps)
    scale = w * inv_std
    shift = b - mean_c * scale

    x_base = x_ptr + pid * HW
    out_base = out_ptr + pid * HW

    sum_val = 0.0

    for block_start in range(0, HW, BLOCK_HW):
        offsets = block_start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW

        # Load x and convert to fp32 for computation
        x = tl.load(x_base + offsets, mask=mask, other=0.0).to(tl.float32)

        # Apply batch norm
        y = x * scale + shift

        # Apply SiLU: y * sigmoid(y)
        y_silu = y * tl.sigmoid(y)

        # Store result (auto-converts to output dtype)
        tl.store(out_base + offsets, y_silu, mask=mask)

        # Accumulate sum for mean computation
        sum_val += tl.sum(tl.where(mask, y_silu, 0.0))

    # Store mean for this (n, c) pair
    tl.store(mean_ptr + pid, sum_val / HW)


@torch.fx.wrap
def bn_silu_mean(x, running_mean, running_var, weight, bias):
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    orig_dtype = x.dtype

    # Output tensor in same dtype as input
    out = torch.empty_like(x)
    # Mean accumulator in float32 for accuracy
    mean_f32 = torch.empty(N * C, device=x.device, dtype=torch.float32)

    grid = (N * C,)
    bn_silu_mean_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out, mean_f32,
        C, HW, 1e-5,
    )

    mean_out = mean_f32.view(N, C, 1, 1).to(orig_dtype)
    return out, mean_out


def replacement_func():
    return bn_silu_mean