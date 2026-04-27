import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements', 'C', 'HW'],
)
@triton.jit
def fused_bn_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and upcast to float32 for numerical stability
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute channel index for shape (1, C, H, W):
    # flat index = c * HW + spatial_idx  => channel = floor(offset / HW) % C
    channel = (offsets // HW) % C

    # Load per-channel batch-norm statistics (float32 on GPU)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + channel, mask=mask, other=1.0)
    w    = tl.load(weight_ptr + channel, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + channel, mask=mask, other=0.0)

    # Batch-norm inference: y = (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * inv_std * w + b

    # SiLU: y * sigmoid(y)
    y = y * tl.sigmoid(y)

    # Store – Triton auto-casts float32 → float16/bfloat16 via pointer dtype
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def fused_bn_silu(x, running_mean, running_var, weight, bias):
    """
    Fused batch-norm (inference) + SiLU.
    x             : (N, C, H, W) tensor on CUDA (float16 or bfloat16)
    running_mean  : (C,) – may be on CPU
    running_var   : (C,) – may be on CPU
    weight        : (C,) – may be on CPU
    bias          : (C,) – may be on CPU
    """
    n_elements = x.numel()
    C  = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    device = x.device

    # Move per-channel statistics to the compute device as float32
    mean_f32 = running_mean.to(device=device, dtype=torch.float32)
    var_f32  = running_var.to(device=device, dtype=torch.float32)
    w_f32    = weight.to(device=device, dtype=torch.float32)
    b_f32    = bias.to(device=device, dtype=torch.float32)

    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    fused_bn_silu_kernel[grid](
        x, mean_f32, var_f32, w_f32, b_f32, out,
        n_elements, C, HW,
        1e-5,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, weight, bias):
    """
    Matches:  batch_norm(training=False, momentum=0.1, eps=1e-05)  followed by
              silu(inplace=True)
    The argument order mirrors the model.py call signature:
        batch_norm(input, running_mean, running_var, weight, bias, ...)
    """
    out = torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )
    out = torch.nn.functional.silu(out, inplace=True)
    return out


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


def replacement_func():
    return fused_bn_silu