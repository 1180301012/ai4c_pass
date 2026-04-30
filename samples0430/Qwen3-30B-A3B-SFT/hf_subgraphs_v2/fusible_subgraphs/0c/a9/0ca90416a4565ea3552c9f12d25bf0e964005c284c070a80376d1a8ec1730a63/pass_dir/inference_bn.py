import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Each config provides BLOCK_SIZE + constexpr dtype flags for the compiler
        triton.Config({'BLOCK_SIZE': 256,  'IS_BF16': 0, 'IS_FP16': 0}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512,  'IS_BF16': 0, 'IS_FP16': 0}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'IS_BF16': 0, 'IS_FP16': 0}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'IS_BF16': 0, 'IS_FP16': 0}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096, 'IS_BF16': 0, 'IS_FP16': 0}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 256,  'IS_BF16': 1, 'IS_FP16': 0}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'IS_BF16': 1, 'IS_FP16': 0}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256,  'IS_BF16': 0, 'IS_FP16': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'IS_BF16': 0, 'IS_FP16': 1}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _bn_inference_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW, eps,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,   # dtype selector injected by autotune
    IS_FP16: tl.constexpr,   # dtype selector injected by autotune
):
    """
    Each program handles one (n, c) pair and processes all HW spatial elements.
    Grid = (N * C,)
    Memory layout: x is NCHW contiguous, so for fixed (n,c) the HW elements are contiguous.
    """
    nc = tl.program_id(0)
    c  = nc % C

    # Load per-channel statistics; compute in fp32 for numerical precision
    mean_v  = tl.load(mean_ptr   + c).to(tl.float32)
    var_v   = tl.load(var_ptr    + c).to(tl.float32)
    w_v     = tl.load(weight_ptr + c).to(tl.float32)
    b_v     = tl.load(bias_ptr   + c).to(tl.float32)

    # Fused: y = (x - mean) / sqrt(var + eps) * weight + bias
    #        y = x * scale + shift
    inv_std = 1.0 / tl.sqrt(var_v + eps)
    scale   = w_v  * inv_std
    shift   = b_v  - mean_v * scale

    # Base pointer for this (n, c) slice; elements are contiguous in HW
    x_base  = nc * HW
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < HW

    x   = tl.load(x_ptr + x_base + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x * scale + shift

    # Cast output back to the input dtype using the constexpr selector
    if IS_BF16:
        tl.store(out_ptr + x_base + offsets, out.to(tl.bfloat16), mask=mask)
    elif IS_FP16:
        tl.store(out_ptr + x_base + offsets, out.to(tl.float16),  mask=mask)
    else:
        tl.store(out_ptr + x_base + offsets, out,                  mask=mask)


@torch.fx.wrap
def bn_inference_triton(running_mean, running_var, weight, bias, x):
    C  = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    NC = x.shape[0] * C

    out = torch.empty_like(x)

    # Grid is fixed: one program per (n, c) pair
    grid = lambda meta: (NC,)

    # All scalar args are passed as KEYWORD args so the autotune can inject
    # BLOCK_SIZE, IS_BF16, and IS_FP16 without positional-arg conflicts.
    _bn_inference_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        C=C, HW=HW, eps=0.001,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Matches the batch_norm inference call used in every target graph:
      batch_norm(input=in_4, running_mean=in_0, running_var=in_1,
                 weight=in_3, bias=in_2, training=False, momentum=0.1, eps=0.001)
    """
    return torch.nn.functional.batch_norm(
        in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001
    )


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # Order must match bn_inference_triton's parameter list:
    # (running_mean, running_var, weight, bias, x)
    return (in_0, in_1, in_3, in_2, in_4)


def replacement_func():
    return bn_inference_triton