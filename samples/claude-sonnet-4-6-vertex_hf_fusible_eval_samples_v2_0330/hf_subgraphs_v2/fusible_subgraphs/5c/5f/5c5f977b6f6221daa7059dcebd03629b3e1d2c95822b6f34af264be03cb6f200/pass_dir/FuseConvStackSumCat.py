"""
Optimization Pass: FuseConvStackSumCat

Pattern:
    conv2d(x, weight, bias, stride=(1,1), pad=(0,0), dilation=(1,1), groups=1)
    → stack([conv_out], dim=0)   ← single-element stack, identity
    → .sum(dim=0)                ← sum of single element, identity
    → cat([sum_out, other], 1)   ← concat along channel dim

Optimization:
    - stack+sum on a single tensor is a mathematical identity; eliminate it entirely.
    - Implement the 1×1 convolution as a Triton GEMM (weight × input per spatial pos).
    - Fuse the cat: pre-allocate the full output buffer and write the conv result and
      the "other" tensor directly into it without any intermediate allocation.

Supported dtypes: float32, float16, bfloat16
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------

def pattern(bias, weight, x, other):
    """
    Matches the subgraph:
        conv2d → stack([conv], dim=0) → sum(dim=0) → cat([sum, other], 1)
    """
    conv = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv], dim=0)
    s = stacked.sum(dim=0)
    result = torch.cat([s, other], 1)
    return result


def replacement_args(bias, weight, x, other):
    return (bias, weight, x, other)


# ---------------------------------------------------------------------------
# Triton kernel 1: 1×1 conv as GEMM → writes into out[:, :C_out, :, :]
#
# The 1×1 conv per (n, hw) is:
#   out[n, co, hw] = sum_ci(weight[co, ci] * input[n, ci, hw]) + bias[co]
#
# We tile over (co, hw) and reduce over ci.
# Grid: (ceil(C_out/BLOCK_CO), ceil(HW/BLOCK_HW), N)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CO': 64,  'BLOCK_HW': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_HW': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_CO': 64,  'BLOCK_HW': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_HW': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_HW': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_HW': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['C_in', 'C_out', 'HW'],
)
@triton.jit
def _conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_in, C_out, HW,
    input_stride_n,   # C_in * HW
    out_stride_n,     # (C_out + C_other) * HW
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_K:  tl.constexpr,
):
    pid_co = tl.program_id(0)
    pid_hw = tl.program_id(1)
    n      = tl.program_id(2)

    co_start = pid_co * BLOCK_CO
    hw_start = pid_hw * BLOCK_HW

    co_offsets = co_start + tl.arange(0, BLOCK_CO)
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    co_mask = co_offsets < C_out
    hw_mask = hw_offsets < HW

    # Float32 accumulator for numerical stability across all dtypes
    acc = tl.zeros([BLOCK_CO, BLOCK_HW], dtype=tl.float32)

    # Reduction over C_in in chunks of BLOCK_K
    for k_start in range(0, C_in, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < C_in

        # Weight tile [BLOCK_CO, BLOCK_K]: weight[co, ci], row-major
        w_ptrs = weight_ptr + co_offsets[:, None] * C_in + k_offsets[None, :]
        # Input tile [BLOCK_K, BLOCK_HW]: input[n, ci, hw]
        # Each row (fixed ci) is contiguous over hw → coalesced loads
        inp_ptrs = (input_ptr + n * input_stride_n
                    + k_offsets[:, None] * HW
                    + hw_offsets[None, :])

        if IS_FP16:
            # Keep FP16 to use FP16 tensor cores (165 TFLOPS on A30)
            w_tile = tl.load(w_ptrs,
                             mask=co_mask[:, None] & k_mask[None, :],
                             other=0.0)
            inp_tile = tl.load(inp_ptrs,
                               mask=k_mask[:, None] & hw_mask[None, :],
                               other=0.0)
            acc += tl.dot(w_tile, inp_tile, out_dtype=tl.float32)
        elif IS_BF16:
            # Keep BF16 to use BF16 tensor cores (165 TFLOPS on A30)
            w_tile = tl.load(w_ptrs,
                             mask=co_mask[:, None] & k_mask[None, :],
                             other=0.0)
            inp_tile = tl.load(inp_ptrs,
                               mask=k_mask[:, None] & hw_mask[None, :],
                               other=0.0)
            acc += tl.dot(w_tile, inp_tile, out_dtype=tl.float32)
        else:
            # FP32 with TF32 tensor cores (41 TFLOPS on A30)
            w_tile = tl.load(w_ptrs,
                             mask=co_mask[:, None] & k_mask[None, :],
                             other=0.0)
            inp_tile = tl.load(inp_ptrs,
                               mask=k_mask[:, None] & hw_mask[None, :],
                               other=0.0)
            acc += tl.dot(w_tile, inp_tile, allow_tf32=True)

    # Add bias (broadcast over HW dimension)
    bias_vals = tl.load(bias_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + bias_vals[:, None]

    # Cast result back to output dtype
    if IS_FP16:
        out_vals = acc.to(tl.float16)
    elif IS_BF16:
        out_vals = acc.to(tl.bfloat16)
    else:
        out_vals = acc  # float32

    # Store into the conv-output slice of the pre-allocated buffer
    out_ptrs = out_ptr + n * out_stride_n + co_offsets[:, None] * HW + hw_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=co_mask[:, None] & hw_mask[None, :])


# ---------------------------------------------------------------------------
# Triton kernel 2: copy "other" → out[:, C_out:, :, :]
#
# For NCHW contiguous tensors the per-batch data is contiguous, so this
# is a strided memcpy with one block-id per BLOCK_SIZE chunk.
# Grid: (ceil(Cb_HW / BLOCK), N)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}),
        triton.Config({'BLOCK': 2048}),
        triton.Config({'BLOCK': 4096}),
        triton.Config({'BLOCK': 8192}),
    ],
    key=['Cb_HW'],
)
@triton.jit
def _copy_other_kernel(
    other_ptr, out_ptr,
    Cb_HW,       # C_other * H * W  (elements per batch in other)
    C_out_HW,    # C_out  * H * W  (byte-offset to where other starts in out)
    out_stride_n,  # (C_out + C_other) * H * W
    BLOCK: tl.constexpr,
):
    pid_x = tl.program_id(0)
    n     = tl.program_id(1)

    start   = pid_x * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask    = offsets < Cb_HW

    vals = tl.load(other_ptr + n * Cb_HW + offsets, mask=mask)
    tl.store(out_ptr + n * out_stride_n + C_out_HW + offsets, vals, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def optimized_conv_stack_sum_cat(bias, weight, x, other):
    """
    Fused replacement for: conv2d → stack([x], 0) → sum(0) → cat([·, other], 1)

    Steps:
      1. Allocate the full output [N, C_out+C_other, H, W] up front.
      2. Fill out[:, :C_out]  with the 1×1 conv result (Triton GEMM),
         using native FP16/BF16 tensor cores or TF32 depending on dtype.
      3. Fill out[:, C_out:]  with 'other' (Triton memcpy).
    """
    N, C_in, H, W = x.shape
    C_out   = weight.shape[0]
    C_other = other.shape[1]
    HW      = H * W

    is_fp16 = (x.dtype == torch.float16)
    is_bf16 = (x.dtype == torch.bfloat16)

    # Pre-allocate full output
    out = torch.empty(N, C_out + C_other, H, W, dtype=x.dtype, device=x.device)

    out_stride_n = (C_out + C_other) * HW
    C_out_HW     = C_out   * HW
    Cb_HW        = C_other * HW

    # -- Kernel 1: 1×1 conv (GEMM) writes directly into out[:, :C_out] ----
    grid_conv = lambda meta: (
        triton.cdiv(C_out, meta['BLOCK_CO']),
        triton.cdiv(HW,    meta['BLOCK_HW']),
        N,
    )
    _conv1x1_kernel[grid_conv](
        x, weight, bias, out,
        N, C_in, C_out, HW,
        C_in * HW,    # input_stride_n
        out_stride_n,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    # -- Kernel 2: copy other into out[:, C_out:] --------------------------
    grid_copy = lambda meta: (triton.cdiv(Cb_HW, meta['BLOCK']), N)
    _copy_other_kernel[grid_copy](
        other, out,
        Cb_HW, C_out_HW,
        out_stride_n,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement factory (zero-argument, returns callable)
# ---------------------------------------------------------------------------

def replacement_func():
    return optimized_conv_stack_sum_cat