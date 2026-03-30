import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    """
    Matches inference-mode batch_norm with momentum=0.1, eps=1e-05.
    x can be any tensor (e.g., output of conv2d).
    """
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _bn_inf_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    C,
    HW,
    eps,
    BLOCK_HW: tl.constexpr,
):
    """
    Grid: (N*C, ceil(HW / BLOCK_HW))
    Each program handles one (n,c) slice and BLOCK_HW spatial positions.
    BN inference: out = x * (gamma / sqrt(var + eps)) + (beta - mean * gamma / sqrt(var+eps))
    All BN arithmetic is done in float32 for numerical stability.
    """
    nc_pid = tl.program_id(0)
    hw_pid = tl.program_id(1)

    c = nc_pid % C

    # Load BN parameters in float32 (they may be fp16/bf16, cast up)
    mean  = tl.load(mean_ptr  + c).to(tl.float32)
    var   = tl.load(var_ptr   + c).to(tl.float32)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta  = tl.load(beta_ptr  + c).to(tl.float32)

    scale = gamma / tl.sqrt(var + eps)
    shift = beta - mean * scale

    hw_offsets = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW
    base = nc_pid * HW

    x = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0)
    # Upcast to float32, apply affine transform, cast back to input dtype
    out = (x.to(tl.float32) * scale + shift).to(x.dtype)
    tl.store(out_ptr + base + hw_offsets, out, mask=mask)


@torch.fx.wrap
def _bn_inference(x, running_mean, running_var, weight, bias):
    """
    Triton-based inference-mode batch normalization.
    Avoids a separate memory round-trip for the affine transform.
    """
    device = x.device
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    # Move BN buffers/params to the same device as x (they may be on CPU)
    mean  = running_mean.to(device=device)
    var   = running_var.to(device=device)
    gamma = weight.to(device=device)
    beta  = bias.to(device=device)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))
    _bn_inf_kernel[grid](
        x, mean, var, gamma, beta, out,
        C, HW, 1e-5,
    )
    return out


def replacement_func():
    return _bn_inference