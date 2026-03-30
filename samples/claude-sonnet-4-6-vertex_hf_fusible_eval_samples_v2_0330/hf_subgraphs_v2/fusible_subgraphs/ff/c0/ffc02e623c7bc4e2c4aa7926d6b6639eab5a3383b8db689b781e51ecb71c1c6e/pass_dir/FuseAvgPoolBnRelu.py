import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=1),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 64}, num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_avgpool_bn_relu_kernel(
    x_ptr,
    rm_ptr,
    rv_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    bc = tl.program_id(0)
    c = bc % C
    base = bc * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    total = tl.sum(x_f32, axis=0)
    mean_val = total / HW
    rm    = tl.load(rm_ptr + c).to(tl.float32)
    rv    = tl.load(rv_ptr + c).to(tl.float32)
    gamma = tl.load(w_ptr  + c).to(tl.float32)
    beta  = tl.load(b_ptr  + c).to(tl.float32)
    normed     = (mean_val - rm) / tl.sqrt(rv + eps)
    result_f32 = tl.maximum(normed * gamma + beta, 0.0)
    tl.store(out_ptr + bc, result_f32.to(x.dtype))


@torch.fx.wrap
def fused_avgpool_bn_relu(x, weight, bias, running_mean, running_var):
    # x: [B, C, H, W]   weight/bias: gamma/beta [C]   running_mean/var: [C]
    B, C, H, W = x.shape
    HW = H * W
    out = torch.empty(B * C, dtype=x.dtype, device=x.device)
    fused_avgpool_bn_relu_kernel[(B * C,)](
        x, running_mean, running_var, weight, bias, out,
        C, HW, eps=1e-5,
    )
    return out.view(B, C, 1, 1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=1),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_avgpool_bn_kernel(
    x_ptr, rm_ptr, rv_ptr, w_ptr, b_ptr, out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    bc = tl.program_id(0)
    c = bc % C
    base = bc * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    total = tl.sum(x_f32, axis=0)
    mean_val = total / HW
    rm    = tl.load(rm_ptr + c).to(tl.float32)
    rv    = tl.load(rv_ptr + c).to(tl.float32)
    gamma = tl.load(w_ptr  + c).to(tl.float32)
    beta  = tl.load(b_ptr  + c).to(tl.float32)
    normed = (mean_val - rm) / tl.sqrt(rv + eps)
    result = normed * gamma + beta
    tl.store(out_ptr + bc, result.to(x.dtype))


@torch.fx.wrap
def fused_avgpool_bn(x, running_mean, running_var, weight, bias):
    B, C, H, W = x.shape
    HW = H * W
    out = torch.empty(B * C, dtype=x.dtype, device=x.device)
    fused_avgpool_bn_kernel[(B * C,)](
        x, running_mean, running_var, weight, bias, out,
        C, HW, eps=1e-5,
    )
    return out.view(B, C, 1, 1)


# ── Pattern: avgpool + batch_norm (anchor = batch_norm) ──────────────────────
# relu(inplace=True) is excluded since Dynamo normalizes its args differently.
# relu still runs after this replacement on the [B,C,1,1] BN output.
def pattern(x, running_mean, running_var, weight, bias):
    pool = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    return torch.nn.functional.batch_norm(
        pool, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_avgpool_bn