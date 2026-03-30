import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001)
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


@triton.autotune(
    configs=[
        # Small BLOCK_HW → more parallelism; good for small NC (e.g. N=1, NC=128)
        triton.Config({'BLOCK_HW': 128},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4, num_stages=2),
        # Larger BLOCK_HW → fewer blocks but more work/block; good for large NC
        triton.Config({'BLOCK_HW': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=3),
    ],
    key=['HW', 'NC'],   # NC = N*C: controls dim-0 grid size → different NC needs different BLOCK_HW
)
@triton.jit
def _bn_prelu_kernel(
    x_ptr, out_ptr,
    running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr,
    prelu_weight_ptr,
    C, HW,
    NC,               # N*C — passed only as an autotune key; not used in the body
    BLOCK_HW: tl.constexpr,
):
    """
    2-D grid: (N*C, ceil(HW / BLOCK_HW))
      pid_nc → which (batch, channel) pair   [dim 0]
      pid_hw → which spatial block           [dim 1]

    Per-channel BN/PReLU params are loaded once per program as scalars —
    no per-element gather, perfectly coalesced spatial loads.
    """
    pid_nc = tl.program_id(0)   # n * C + c
    pid_hw = tl.program_id(1)   # spatial-block index

    c = pid_nc % C              # channel index (computed once per program)

    # ---- Scalar loads of per-channel parameters ----
    mean    = tl.load(running_mean_ptr + c).to(tl.float32)
    var     = tl.load(running_var_ptr  + c).to(tl.float32)
    bn_w    = tl.load(bn_weight_ptr    + c).to(tl.float32)
    bn_b    = tl.load(bn_bias_ptr      + c).to(tl.float32)
    prelu_w = tl.load(prelu_weight_ptr + c).to(tl.float32)

    # Precompute fused BN coefficients: y = scale * x + offset
    eps     = 0.001
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = bn_w * inv_std
    offset  = bn_b - bn_w * mean * inv_std

    # ---- Vectorised load / compute / store over HW block ----
    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = hw_off < HW
    base   = pid_nc * HW

    x_raw  = tl.load(x_ptr + base + hw_off, mask=mask, other=0.0)
    x_fp32 = x_raw.to(tl.float32)

    # Batch Norm (inference)
    y = scale * x_fp32 + offset

    # PReLU
    z = tl.where(y >= 0.0, y, prelu_w * y)

    # Store — preserve original dtype
    tl.store(out_ptr + base + hw_off, z.to(x_raw.dtype), mask=mask)


@torch.fx.wrap
def bn_prelu_fused(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    """
    Fused BatchNorm (inference) + PReLU.
    Returns a tensor of the same shape and dtype as x.
    """
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out  = torch.empty_like(x)
    grid = lambda meta: (NC, triton.cdiv(HW, meta['BLOCK_HW']))

    _bn_prelu_kernel[grid](
        x, out,
        running_mean, running_var,
        bn_weight, bn_bias, prelu_weight,
        C, HW, NC,
    )

    return out


def replacement_func():
    return bn_prelu_fused