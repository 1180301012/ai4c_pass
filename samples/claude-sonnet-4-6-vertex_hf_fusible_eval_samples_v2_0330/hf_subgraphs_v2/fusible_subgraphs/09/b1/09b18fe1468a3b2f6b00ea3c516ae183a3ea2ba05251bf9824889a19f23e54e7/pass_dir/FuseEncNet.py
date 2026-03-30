"""
FuseEncNet: Fuses two independent subgraphs in the EncNet encoding step.

Pattern (no softmax — avoids decomposition issues):
  Chain A — distance computation (tmp_4):
      tmp_1 = in_1 - in_2          [1,4096,32,512]
      tmp_2 = tmp_1.pow(2)         [1,4096,32,512]
      tmp_3 = tmp_2.sum(dim=3)     [1,4096,32]
      tmp_4 = in_3 * tmp_3         [1,4096,32]   ← output A
  Chain B — broadcast difference (tmp_10):
      tmp_7  = in_4.unsqueeze(2)
      tmp_8  = tmp_7.expand((1,4096,32,512))
      tmp_10 = tmp_8 - tmp_6       [1,4096,32,512] ← output B

Returns (tmp_10, tmp_4).

Key design:
  * dist_kernel replicates PyTorch's float16 path exactly:
      diff  = in1 - in2   (float16)
      sq    = diff * diff  (float16)
      total = sum(sq→f32)  (float32 accumulation, same as PyTorch)
      result= total.to(f16) * scale.to(f16)  (float16)
  * replacement_func returns a non-@wrapped helper that uses getitem so the
    framework sees 2 separate returning nodes matching the pattern's 2 outputs.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern — all 5 model inputs, includes in_0.view so all args are placeholders
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_1  = in_1 - in_2
    tmp_2  = tmp_1.pow(2)
    tmp_3  = tmp_2.sum(dim = 3)
    tmp_4  = in_3 * tmp_3
    tmp_6  = in_0.view((1, 1, 32, 512))
    tmp_7  = in_4.unsqueeze(2)
    tmp_8  = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return (tmp_10, tmp_4)


# ---------------------------------------------------------------------------
# Kernel A: squared-distance + scale — exactly matches PyTorch float16 ops
#   Computes: diff=in1-in2 (f16), sq=diff^2 (f16), sum→f32, * scale (f16) → f16
# ---------------------------------------------------------------------------

@triton.jit
def dist_kernel(
    in1_ptr, in2_ptr, in3_ptr, out_ptr,
    N, K, D,
    BLOCK_D: tl.constexpr,
    IS_FLOAT16: tl.constexpr,
):
    pid    = tl.program_id(0)    # = n * K + k
    n      = pid // K
    k      = pid % K
    d_offs = tl.arange(0, BLOCK_D)

    # Load in native dtype (float16 or bfloat16)
    x    = tl.load(in1_ptr + n * K * D + k * D + d_offs)
    c    = tl.load(in2_ptr +             k * D + d_offs)

    diff = x - c            # native dtype — matches PyTorch tmp_1
    sq   = diff * diff       # native dtype — matches PyTorch tmp_2

    # Float32 accumulation — matches PyTorch's float16/bfloat16 sum behavior
    total = tl.sum(sq.to(tl.float32), axis=0)    # fp32 scalar

    # Convert total back to native dtype and multiply by scale in native dtype
    # (exact match with PyTorch's: tmp_4 = in_3 * tmp_3, both in native dtype)
    scale = tl.load(in3_ptr + k)                  # native dtype
    if IS_FLOAT16:
        result = total.to(tl.float16) * scale      # f16 × f16 = f16
    else:
        result = total.to(tl.bfloat16) * scale     # bf16 × bf16 = bf16

    tl.store(out_ptr + pid, result)                # native dtype store


# ---------------------------------------------------------------------------
# Kernel B: broadcast difference (native dtype)
# ---------------------------------------------------------------------------

@triton.jit
def diff_kernel(
    x_ptr, y_ptr, out_ptr,
    N, K, D,
    BLOCK_D: tl.constexpr,
):
    pid    = tl.program_id(0)
    n      = pid // K
    k      = pid % K
    d_offs = tl.arange(0, BLOCK_D)

    xv = tl.load(x_ptr + n * D + d_offs)
    yv = tl.load(y_ptr + k * D + d_offs)

    tl.store(out_ptr + pid * D + d_offs, xv - yv)


# ---------------------------------------------------------------------------
# Merged kernel: computes both chains in one launch to eliminate overhead.
# ---------------------------------------------------------------------------

@triton.jit
def merged_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, in4_ptr,
    out_dist_ptr, out_diff_ptr,
    N, K, D,
    BLOCK_D: tl.constexpr,
    IS_FLOAT16: tl.constexpr,
):
    pid    = tl.program_id(0)    # = n * K + k
    n      = pid // K
    k      = pid % K
    d_offs = tl.arange(0, BLOCK_D)

    # --- Chain A: scaled squared distance ---
    x1   = tl.load(in1_ptr + n * K * D + k * D + d_offs)
    x2   = tl.load(in2_ptr +             k * D + d_offs)
    diff1 = x1 - x2
    sq    = diff1 * diff1
    total = tl.sum(sq.to(tl.float32), axis=0)
    scale = tl.load(in3_ptr + k)
    if IS_FLOAT16:
        dist = total.to(tl.float16) * scale
    else:
        dist = total.to(tl.bfloat16) * scale
    tl.store(out_dist_ptr + pid, dist)

    # --- Chain B: broadcast difference ---
    x4   = tl.load(in4_ptr + n * D + d_offs)
    x0   = tl.load(in0_ptr + k * D + d_offs)
    tl.store(out_diff_ptr + pid * D + d_offs, x4 - x0)


# ---------------------------------------------------------------------------
# Fused kernel wrapper  (opaque to FX via @torch.fx.wrap)
# Returns (tmp_10, tmp_4) as a tuple from a single node.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fused_impl(in_0, in_1, in_2, in_3, in_4):
    N      = in_1.shape[1]   # 4096
    K      = in_1.shape[2]   # 32
    D      = in_1.shape[3]   # 512
    device = in_1.device
    dtype  = in_1.dtype

    # Allocate output buffers
    tmp_4_buf  = torch.empty(N * K,     dtype=dtype, device=device)
    tmp_10_buf = torch.empty(N * K * D, dtype=dtype, device=device)

    # Single merged kernel: computes dist + diff in one launch
    merged_kernel[(N * K,)](
        in_0, in_1, in_2, in_3, in_4,
        tmp_4_buf, tmp_10_buf,
        N, K, D,
        BLOCK_D=512,
        IS_FLOAT16=(dtype == torch.float16),
        num_warps=1,
    )
    tmp_4  = tmp_4_buf.view(1, N, K)
    tmp_10 = tmp_10_buf.view(1, N, K, D)

    return (tmp_10, tmp_4)


# ---------------------------------------------------------------------------
# Non-wrapped helper: unpacks the tuple via getitem so the framework sees
# 2 separate "returning nodes" matching the pattern's 2 outputs.
# ---------------------------------------------------------------------------

def _replacement_wrapper(in_0, in_1, in_2, in_3, in_4):
    result = _fused_impl(in_0, in_1, in_2, in_3, in_4)
    return result[0], result[1]


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return _replacement_wrapper