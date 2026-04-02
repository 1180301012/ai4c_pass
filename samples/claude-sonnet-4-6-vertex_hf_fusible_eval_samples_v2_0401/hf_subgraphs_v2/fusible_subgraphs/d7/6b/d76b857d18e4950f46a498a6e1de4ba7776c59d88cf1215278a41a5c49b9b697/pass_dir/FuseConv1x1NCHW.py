"""
Optimization pass: replace 1×1 conv2d (stride=1, pad=0, dil=1, groups=1)
with a custom Triton GEMM kernel that handles NCHW tensors in-place.

Why this can beat cuDNN for the specific [B, 512, 64, 64] → [B, 21, 64, 64]
shape (weight [21, 512, 1, 1]):

1.  Weight matrix fits in L2 cache (21×512×4B = 42 KB << A30's 48 MB L2).
    After the first few M-tiles, all subsequent tiles read weight from L2,
    effectively loading weight from DRAM only ONCE.

2.  BLOCK_N = 32 (padded from 21) satisfies Tensor Core alignment (≥16),
    allowing tl.dot to use fp16/tf32 Tensor Cores.  cuBLAS / cuDNN may fall
    back to scalar GEMM for N=21 (not a multiple of 16 in some code paths).

3.  The GEMM is memory-bandwidth limited (not compute limited) for large M,
    so the combined L2-caching of weights + efficient DRAM streaming of input
    can reduce total DRAM traffic compared to a naive implementation.

Kernel design:
  • A (input)  : [B, C_in, H, W] read directly in NCHW layout
  • B (weight) : [C_out, C_in] = weight[:, :, 0, 0]  (transposed for GEMM)
  • C (output) : [B, C_out, H, W] written in NCHW layout
  • Accumulator: always float32 for numerical stability

NCHW index math:
  m = b*H*W + hw  (spatial index)
  input [m, k]  = b*C_in*H*W + k*H*W + hw
  output[m, n]  = b*C_out*H*W + n*H*W + hw
  weight_T[k,n] = n*C_in + k   (weight stored [C_out, C_in])
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(input_tensor, weight, bias):
    return torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # (BLOCK_M, BLOCK_K, BLOCK_N) – BLOCK_N=32 pads C_out=21 to TC boundary
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32, "BLOCK_N": 32},
                      num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_N": 32},
                      num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_K": 32, "BLOCK_N": 32},
                      num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_K": 64, "BLOCK_N": 32},
                      num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32, "BLOCK_N": 32},
                      num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32, "BLOCK_N": 32},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 64, "BLOCK_N": 32},
                      num_warps=8, num_stages=3),
    ],
    key=["M", "C_in", "C_out", "HW"],
)
@triton.jit
def _conv1x1_nchw_kernel(
    input_ptr,   # [B, C_in, H, W]
    weight_ptr,  # [C_out, C_in]  (squeezed from [C_out, C_in, 1, 1])
    bias_ptr,    # [C_out]
    output_ptr,  # [B, C_out, H, W]
    M,           # B * H * W
    C_in,
    C_out,
    HW,          # H * W
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,   # tl.float32 / tl.float16 / tl.bfloat16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Decompose m into (batch, spatial) for NCHW addressing
    b_offs  = m_offs // HW   # [BLOCK_M]
    hw_offs = m_offs % HW    # [BLOCK_M]

    m_mask = m_offs < M
    n_mask = n_offs < C_out

    # fp32 accumulator (correct even for fp16/bf16 inputs)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(C_in, BLOCK_K)):
        k_start = k * BLOCK_K
        k_offs  = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        k_mask  = k_offs < C_in

        # ---- Load input tile A [BLOCK_M, BLOCK_K] ----
        # input[b, k_idx, hw] → index = b*C_in*HW + k_idx*HW + hw
        a_idx = (b_offs[:, None] * C_in + k_offs[None, :]) * HW + hw_offs[:, None]
        a = tl.load(
            input_ptr + a_idx,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # ---- Load weight tile B [BLOCK_K, BLOCK_N] ----
        # weight stored [C_out, C_in]; weight_T[k,n] = weight[n,k] = n*C_in + k
        b_idx = n_offs[None, :] * C_in + k_offs[:, None]
        b = tl.load(
            weight_ptr + b_idx,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )

        # ---- Tensor-Core GEMM accumulate ----
        acc = tl.dot(a, b, acc)

    # ---- Add bias [C_out] ----
    bias_vals = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc += bias_vals[None, :].to(tl.float32)

    # ---- Store output [BLOCK_M, BLOCK_N] → NCHW ----
    # output[b, n_idx, hw] → index = b*C_out*HW + n_idx*HW + hw
    out_idx = (b_offs[:, None] * C_out + n_offs[None, :]) * HW + hw_offs[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]

    # Cast float32 accumulator to the requested output dtype
    tl.store(output_ptr + out_idx, acc.to(OUTPUT_DTYPE), mask=out_mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fast_conv1x1_nchw(input_tensor, weight, bias):
    B, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    M   = B * H * W
    HW  = H * W

    # Allocate output with same dtype/device as input
    output = torch.empty(
        (B, C_out, H, W),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # Weight is [C_out, C_in, 1, 1] → contiguous view [C_out, C_in]
    weight_2d = weight.view(C_out, C_in)

    # Map torch dtype → tl dtype for the output store
    _dtype_map = {
        torch.float32:  tl.float32,
        torch.float16:  tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    out_dtype = _dtype_map[input_tensor.dtype]

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(C_out, meta["BLOCK_N"]),
    )

    _conv1x1_nchw_kernel[grid](
        input_tensor, weight_2d, bias, output,
        M, C_in, C_out, HW,
        OUTPUT_DTYPE=out_dtype,
    )

    return output


# ---------------------------------------------------------------------------
# replacement_args / replacement_func
# ---------------------------------------------------------------------------

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)


def replacement_func():
    return _fast_conv1x1_nchw