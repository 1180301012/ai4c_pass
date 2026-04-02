import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    bn = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    result = torch.nn.functional.relu(bn, inplace=False)
    return result


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def bn_relu_fused_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid dim 0 = channel index, Grid dim 1 = spatial block index
    c_idx   = tl.program_id(0)
    hw_pid  = tl.program_id(1)

    hw_start   = hw_pid * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask    = hw_offsets < HW

    # ---- Load per-channel BN parameters (scalar, cast to f32) ----
    mean   = tl.load(mean_ptr   + c_idx).to(tl.float32)
    var    = tl.load(var_ptr    + c_idx).to(tl.float32)
    w      = tl.load(weight_ptr + c_idx).to(tl.float32)
    b      = tl.load(bias_ptr   + c_idx).to(tl.float32)

    # Precompute affine coefficients (same for every element in this channel)
    #   y = (x - mean) / sqrt(var + eps) * w + b
    #     = x * scale + shift
    eps   = 1e-05
    scale = w / tl.sqrt(var + eps)        # w / std
    shift = b - mean * scale              # b - mean * w / std

    # ---- Load input elements ----
    x_offsets = c_idx * HW + hw_offsets
    x_raw = tl.load(x_ptr + x_offsets, mask=hw_mask, other=0.0)
    x_f32 = x_raw.to(tl.float32)

    # ---- Fused BN + ReLU ----
    y   = x_f32 * scale + shift
    out = tl.maximum(y, 0.0)

    # ---- Store (cast back to original dtype) ----
    tl.store(out_ptr + x_offsets, out.to(x_raw.dtype), mask=hw_mask)


@torch.fx.wrap
def bn_relu_triton(x, running_mean, running_var, weight, bias):
    # x is expected to be [N, C, H, W] with N=1
    C  = x.shape[1]
    HW = x.shape[2] * x.shape[3]

    out = torch.empty_like(x)

    grid = lambda meta: (C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    bn_relu_fused_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        HW=HW,
    )

    return out


def replacement_func():
    return bn_relu_triton