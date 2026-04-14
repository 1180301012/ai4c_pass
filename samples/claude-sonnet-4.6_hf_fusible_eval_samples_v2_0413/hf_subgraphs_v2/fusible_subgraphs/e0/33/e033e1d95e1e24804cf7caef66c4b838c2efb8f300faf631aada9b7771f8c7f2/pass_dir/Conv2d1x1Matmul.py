import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════════════
# Triton replacement for torch.conv2d with 1×1 kernel, stride=1, pad=0.
#
# cuDNN for NCHW-format 1×1 conv:  NCHW→NHWC copy + GEMM + NHWC→NCHW copy + overhead
# Our kernel:  single launch, direct NCHW addressing, no format conversion.
#
# Formulation:  out[b] = W_2d @ in3[b].view(C_in, H*W) + bias
#   W_2d = in_1.view(C_out, C_in)  [C_out × C_in, row-major, stride C_in per row]
#   in3[b] viewed as [C_in, H*W]:  in3[b,k,hw] = in3_ptr + b*C_IN*HW + k*HW + hw
#   → both are contiguous in their innermost dimension, no copies needed.
#
# Tile sizes (standard Triton matmul tutorial):
#   BLOCK_M=128 (C_out), BLOCK_N=128 (H*W), BLOCK_K=32 (C_in), num_warps=8
#   Larger tiles → higher tensor-core utilisation on Ampere (fp16 mma 16×16×16).
# ══════════════════════════════════════════════════════════════════════════════

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 32


@triton.jit
def _conv1x1_kernel(
    in3_ptr,   # [B, C_IN, HW]   activation
    w_ptr,     # [C_OUT, C_IN]   weight (squeezed 4D→2D)
    bias_ptr,  # [C_OUT]
    out_ptr,   # [B, C_OUT, HW]  output
    B,
    C_IN:    tl.constexpr,
    C_OUT:   tl.constexpr,
    HW:      tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(0)   # batch
    m_idx = tl.program_id(1)   # C_out tile
    n_idx = tl.program_id(2)   # HW tile

    m_off = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M] C_out indices
    n_off = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N] HW indices

    # fp32 accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K-loop over C_in; num_stages pipelines loads to hide HBM latency
    for k in range(0, C_IN, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)

        # Weight tile [BLOCK_M, BLOCK_K]: w[c_out, c_in] = w_ptr + c_out*C_IN + c_in
        w_tile = tl.load(w_ptr + m_off[:, None] * C_IN + k_off[None, :])

        # Input tile [BLOCK_K, BLOCK_N]: in3[b,k,hw] = in3_ptr + b*C_IN*HW + k*HW + hw
        in3_tile = tl.load(in3_ptr + b_idx * C_IN * HW
                                    + k_off[:, None] * HW
                                    + n_off[None, :])

        # Tensor-core GEMM with fp32 accumulation
        acc = tl.dot(w_tile, in3_tile, acc=acc, out_dtype=tl.float32, allow_tf32=True)

    # Bias [BLOCK_M] broadcast over BLOCK_N
    bias = tl.load(bias_ptr + m_off)
    acc += bias[:, None].to(tl.float32)

    # Store: out[b, c_out, hw] = out_ptr + b*C_OUT*HW + c_out*HW + hw
    tl.store(out_ptr + b_idx * C_OUT * HW + m_off[:, None] * HW + n_off[None, :], acc)


@torch.fx.wrap
def triton_conv2d_1x1(in_3, in_1, in_0):
    """
    Triton replacement for:
      torch.conv2d(in_3, in_1, in_0, (1,1), (0,0), (1,1), 1)
    """
    B     = in_3.shape[0]
    C_in  = in_3.shape[1]   # 512
    H     = in_3.shape[2]   # 64
    W     = in_3.shape[3]   # 64
    C_out = in_1.shape[0]   # 256
    HW    = H * W            # 4096

    output = torch.empty((B, C_out, H, W), dtype=in_3.dtype, device=in_3.device)

    # Grid: batch × (C_out / BLOCK_M) × (HW / BLOCK_N)
    # For C_out=256, HW=4096: 2 × 32 = 64 programs per batch
    grid = (B, C_out // _BLOCK_M, HW // _BLOCK_N)

    _conv1x1_kernel[grid](
        in_3, in_1, in_0, output,
        B,
        C_IN=C_in, C_OUT=C_out, HW=HW,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        num_warps=8,    # 256 threads; matches BLOCK_M=128 for register balance
        num_stages=3,   # software-pipeline K-loop: 3 tile pairs in flight
    )

    return output


# ── Pass API ───────────────────────────────────────────────────────────────────

def pattern(in_3, in_1, in_0):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)


def replacement_func():
    return triton_conv2d_1x1