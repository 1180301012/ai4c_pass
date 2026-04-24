import torch
import triton
import triton.language as tl


def pattern(in_4, in_0, in_1, in_3, in_2):
    """
    Match inference-mode batch norm (training=False, momentum=0.1, eps=0.001).
    Arguments mirror model.py exactly:
      in_4: input activation, in_0: running_mean, in_1: running_var,
      in_3: weight (gamma), in_2: bias (beta)
    """
    return torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)


def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)


# -----------------------------------------------------------------------
# Triton kernel: inference-mode batch norm, 3D grid (C, N, HW_BLOCKS).
# Grid dim-2 tiles the spatial dimension for more GPU parallelism.
# eviction_policy="evict_first" avoids L2 pollution from the streaming
# input data (only the tiny channel params stay in cache).
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64},   num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 128},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 256},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=4),
    ],
    key=['N', 'C', 'HW'],   # separate tuning per (batch, channels, spatial)
)
@triton.jit
def _bn_inference_3d_kernel(
    input_ptr,   # [N, C, H, W] contiguous fp32/fp16/bf16
    mean_ptr,    # [C]
    var_ptr,     # [C]
    weight_ptr,  # [C]
    bias_ptr,    # [C]
    output_ptr,  # [N, C, H, W] same dtype
    N, C, HW,
    BLOCK: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # 3D grid: dim-0=C (channel), dim-1=N (batch), dim-2=HW block
    c      = tl.program_id(0)
    n      = tl.program_id(1)
    hw_blk = tl.program_id(2)

    hw_start = hw_blk * BLOCK
    base     = (n * C + c) * HW + hw_start

    # Per-channel BN params – 4 scalar loads, always in L1/L2 cache
    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    var    = tl.load(var_ptr    + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias   = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + 0.001)
    scale   = weight * inv_std
    offset  = bias - mean * scale

    # Streaming load: evict_first avoids polluting L2 with the large input tensor
    local_offs = tl.arange(0, BLOCK)
    hw_offs    = hw_start + local_offs
    mask       = hw_offs < HW

    x      = tl.load(input_ptr + base + local_offs, mask=mask, other=0.0,
                     eviction_policy="evict_first").to(tl.float32)
    result = x * scale + offset

    if IS_BF16:
        tl.store(output_ptr + base + local_offs, result.to(tl.bfloat16), mask=mask)
    elif IS_FP16:
        tl.store(output_ptr + base + local_offs, result.to(tl.float16),  mask=mask)
    else:
        tl.store(output_ptr + base + local_offs, result, mask=mask)


# -----------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------
@torch.fx.wrap
def batch_norm_inference_triton(in_4, in_0, in_1, in_3, in_2):
    """
    Triton-accelerated batch-norm inference (training=False, eps=0.001).

    in_4 : input  [N, C, H, W]
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_3 : weight/gamma  [C]
    in_2 : bias/beta     [C]
    """
    device = in_4.device

    running_mean = torch.as_tensor(in_0, device=device)
    running_var  = torch.as_tensor(in_1, device=device)
    weight       = torch.as_tensor(in_3, device=device)
    bias         = torch.as_tensor(in_2, device=device)

    N, C, H, W = in_4.shape
    HW = H * W

    output = torch.empty_like(in_4)

    is_fp16 = (in_4.dtype == torch.float16)
    is_bf16 = (in_4.dtype == torch.bfloat16)

    # 3D grid: dim-2 tiles HW for more GPU parallelism (especially helpful for small N)
    grid = lambda meta: (C, N, triton.cdiv(HW, meta['BLOCK']))

    _bn_inference_3d_kernel[grid](
        in_4,
        running_mean, running_var, weight, bias,
        output,
        N, C, HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    return output


def replacement_func():
    return batch_norm_inference_triton