import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['N_elem'],
)
@triton.jit
def fused_bn_lrelu_add_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    N_elem,
    HW,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elem

    # Channel index for [N, C, H, W] layout
    channel_idx = (offsets // HW) % C

    # Load inputs (in original dtype)
    x        = tl.load(x_ptr        + offsets,      mask=mask, other=0.0)
    residual = tl.load(residual_ptr  + offsets,      mask=mask, other=0.0)

    # Load BN parameters (per-channel) and cast to float32 for precision
    mean = tl.load(running_mean_ptr + channel_idx,  mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(running_var_ptr  + channel_idx,  mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr       + channel_idx,  mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr         + channel_idx,  mask=mask, other=0.0).to(tl.float32)

    # Batch-norm inference in float32
    x_f32  = x.to(tl.float32)
    x_norm = (x_f32 - mean) * tl.rsqrt(var + 1e-5)
    y_f32  = x_norm * w + b

    # LeakyReLU (negative slope = 0.01)
    y_f32 = tl.where(y_f32 >= 0.0, y_f32, y_f32 * 0.01)

    # Cast back to input dtype and add residual
    y   = y_f32.to(x.dtype)
    out = y + residual

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_bn_lrelu_add(x, running_mean, running_var, weight, bias, residual):
    N  = x.shape[0]
    C  = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    N_elem = N * C * HW

    # BN params may live on CPU; move them to the same device as x
    device = x.device
    running_mean = running_mean.to(device)
    running_var  = running_var.to(device)
    weight       = weight.to(device)
    bias         = bias.to(device)

    # Ensure contiguous layout
    x        = x.contiguous()
    residual = residual.contiguous()

    out = torch.empty_like(x)

    grid = lambda meta: ((N_elem + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_bn_lrelu_add_kernel[grid](
        x, running_mean, running_var, weight, bias, residual, out,
        N_elem, HW, C,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, weight, bias, residual):
    tmp_6 = torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + residual
    return tmp_8


def replacement_args(x, running_mean, running_var, weight, bias, residual):
    return (x, running_mean, running_var, weight, bias, residual)


def replacement_func():
    return fused_bn_lrelu_add