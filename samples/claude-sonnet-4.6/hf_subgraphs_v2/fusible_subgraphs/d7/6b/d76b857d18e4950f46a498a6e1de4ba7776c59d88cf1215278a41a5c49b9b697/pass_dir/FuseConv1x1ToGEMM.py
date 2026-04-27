import torch
import triton
import triton.language as tl


# Match the 1x1 conv exactly as it appears in model.py (positional args)
def pattern(in_0, in_1, in_2):
    return torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# -----------------------------------------------------------------------
# 1x1 Conv as batched GEMM:
#   A = weight  [C_out, C_in]          (small, fits in L2)
#   B = input_b [C_in,  HW]            (large, streamed from HBM)
#   C = output_b[C_out, HW] + bias
#
# Memory layout (NCHW):
#   input [b, c, h, w]  -> input_ptr + b*(C_in*HW) + c*HW + (h*W+w)
#   weight[m, k, 0, 0]  -> weight_ptr + m*C_in + k   (1x1 so last 2 dims =1)
#   output[b, m, h, w]  -> output_ptr + b*(C_out*HW) + m*HW + (h*W+w)
# -----------------------------------------------------------------------
@triton.jit
def _conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_out, C_in, HW,
    BLOCK_M: tl.constexpr,   # tile along C_out  (32)
    BLOCK_N: tl.constexpr,   # tile along HW     (64)
    BLOCK_K: tl.constexpr,   # tile along C_in   (32)
    OUT_DTYPE_ID: tl.constexpr,  # 0=f32, 1=f16, 2=bf16
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(HW, BLOCK_N)
    num_m_blocks = tl.cdiv(C_out, BLOCK_M)

    b_idx  = pid // (num_m_blocks * num_n_blocks)
    mn_idx = pid %  (num_m_blocks * num_n_blocks)
    m_blk  = mn_idx // num_n_blocks
    n_blk  = mn_idx %  num_n_blocks

    m_off = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_off < C_out
    n_mask = n_off < HW

    # Float32 accumulator (precision for all dtypes)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C_in, BLOCK_K):
        k_off  = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < C_in

        # Weight tile  [BLOCK_M, BLOCK_K] — row-major, col stride = 1
        a_ptrs = weight_ptr + m_off[:, None] * C_in + k_off[None, :]
        a = tl.load(a_ptrs,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0)

        # Input tile   [BLOCK_K, BLOCK_N] — row-major (channel, spatial)
        b_ptrs = (input_ptr
                  + b_idx * (C_in * HW)
                  + k_off[:, None] * HW
                  + n_off[None, :])
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0)

        # Mixed-precision matmul: accumulate into float32
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    # Bias broadcast
    bias = tl.load(bias_ptr + m_off, mask=m_mask, other=0.0)
    acc += bias.to(tl.float32)[:, None]

    # Store with dtype conversion
    out_ptrs = (output_ptr
                + b_idx * (C_out * HW)
                + m_off[:, None] * HW
                + n_off[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]

    if OUT_DTYPE_ID == 1:
        tl.store(out_ptrs, acc.to(tl.float16),   mask=out_mask)
    elif OUT_DTYPE_ID == 2:
        tl.store(out_ptrs, acc.to(tl.bfloat16),  mask=out_mask)
    else:
        tl.store(out_ptrs, acc,                   mask=out_mask)


@torch.fx.wrap
def triton_conv1x1(in_0_bias, in_1_weight, in_2_input):
    B, C_in, H, W = in_2_input.shape
    C_out = in_0_bias.shape[0]
    HW    = H * W

    output = torch.empty((B, C_out, H, W),
                          dtype=in_2_input.dtype,
                          device=in_2_input.device)

    _dtype_id = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
    OUT_DTYPE_ID = _dtype_id.get(in_2_input.dtype, 0)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    num_m_blocks = (C_out + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (HW   + BLOCK_N - 1) // BLOCK_N
    grid = (B * num_m_blocks * num_n_blocks,)

    _conv1x1_kernel[grid](
        in_2_input, in_1_weight, in_0_bias, output,
        B, C_out, C_in, HW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        OUT_DTYPE_ID=OUT_DTYPE_ID,
        num_warps=4,
        num_stages=2,
    )

    return output


def replacement_func():
    return triton_conv1x1