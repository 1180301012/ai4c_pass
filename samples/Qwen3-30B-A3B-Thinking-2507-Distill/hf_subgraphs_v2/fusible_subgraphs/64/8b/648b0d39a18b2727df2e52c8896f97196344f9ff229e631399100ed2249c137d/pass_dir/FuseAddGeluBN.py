import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Per-channel inference BN kernel for NCHW tensors.
# Grid dim 0 = NC (one CTA per (n,c) pair).
# Grid dim 1 = ceil(HW / BLOCK_SIZE) (tiles the spatial dimension).
# Per element: out = (x - mean[c]) / sqrt(var[c]+eps) * weight[c] + bias[c]
# ---------------------------------------------------------------------------

@triton.jit
def _bn_kernel(
    x_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    nc   = tl.program_id(0)
    hw_b = tl.program_id(1)
    c    = nc % C

    # Per-channel BN params in fp32
    mean_c   = tl.load(mean_ptr   + c).to(tl.float32)
    var_c    = tl.load(var_ptr    + c).to(tl.float32)
    weight_c = tl.load(weight_ptr + c).to(tl.float32)
    bias_c   = tl.load(bias_ptr   + c).to(tl.float32)

    eps     = 1e-5
    inv_std = 1.0 / tl.sqrt(var_c + eps)
    scale   = weight_c * inv_std
    shift   = bias_c - mean_c * scale

    base    = nc * HW + hw_b * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = (hw_b * BLOCK_SIZE + offsets) < HW

    x       = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    out_val = x * scale + shift

    tl.store(out_ptr + base + offsets, out_val, mask=mask)


@torch.fx.wrap
def triton_bn_inference(x, mean, var, weight, bias):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty_like(x)

    BLOCK_SIZE    = 1024
    num_hw_blocks = (HW + BLOCK_SIZE - 1) // BLOCK_SIZE

    _bn_kernel[(NC, num_hw_blocks)](
        x,
        mean, var, weight, bias,
        out,
        C, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern: match batch_norm(x, mean, var, weight, bias, training=False)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, x):
    return torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)


def replacement_args(in_0, in_1, in_2, in_3, x):
    # in_0=mean, in_1=var, in_2=bias, in_3=weight
    return (x, in_0, in_1, in_3, in_2)


def replacement_func():
    return triton_bn_inference