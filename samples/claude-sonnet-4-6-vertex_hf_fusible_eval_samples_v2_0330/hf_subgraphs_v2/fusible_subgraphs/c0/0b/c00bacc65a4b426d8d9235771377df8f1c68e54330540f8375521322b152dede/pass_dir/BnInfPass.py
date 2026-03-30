import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.jit
def _bn_inf_fwd(
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
    nc_pid = tl.program_id(0)
    hw_pid = tl.program_id(1)

    c = nc_pid % C

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
    out = (x.to(tl.float32) * scale + shift).to(x.dtype)
    tl.store(out_ptr + base + hw_offsets, out, mask=mask)


@torch.fx.wrap
def _bn_inference(x, running_mean, running_var, weight, bias):
    device = x.device
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    mean  = running_mean.to(device=device)
    var   = running_var.to(device=device)
    gamma = weight.to(device=device)
    beta  = bias.to(device=device)

    BLOCK_HW = 256
    grid = (N * C, triton.cdiv(HW, BLOCK_HW))
    _bn_inf_fwd[grid](
        x, mean, var, gamma, beta, out,
        C, HW, 1e-5,
        BLOCK_HW=BLOCK_HW,
    )
    return out


def replacement_func():
    return _bn_inference