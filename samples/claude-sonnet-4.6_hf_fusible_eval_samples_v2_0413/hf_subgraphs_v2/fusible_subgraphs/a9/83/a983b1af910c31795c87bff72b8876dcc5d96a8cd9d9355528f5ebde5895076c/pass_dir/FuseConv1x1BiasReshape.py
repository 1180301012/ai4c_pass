import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match: conv2d (1x1, stride=1, pad=0) -> multiply by 1.0 -> reshape(-1, 17, 4096)
    in_0: bias [C_out=17]
    in_1: weight [C_out=17, C_in=256, 1, 1]
    in_2: input [N, C_in=256, H=64, W=64]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# -----------------------------------------------------------------------
# Key insight for C_out=17:
#   - BLOCK_M=32 covers all 17 channels in ONE tile (17 valid + 15 padded).
#   - This means each input chunk is read ONCE, not twice.
#   - With BLOCK_M=16, we'd need 2 tiles and read input 2x (100% read amplification).
#   - BLOCK_M=32 eliminates that amplification at the cost of 47% wasted FP compute
#     in the padded rows — but FP compute is cheap; memory bandwidth is the bottleneck.
#
# Grid: (ceil(C_out/BLOCK_M), ceil(N*HW/BLOCK_N))
#   - Merging N and HW into one axis avoids 3-D grid launch overhead and gives
#     more programs for small-N cases (better SM utilization).
# -----------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- BLOCK_M=32: one C_out tile covers all 17 channels → no read amplification ----
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 32},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32},  num_warps=4, num_stages=4),
        # ---- BLOCK_M=64: overkill for 17, but may use larger tensor-core tiles ----
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 32},  num_warps=8, num_stages=4),
    ],
    key=['NHW', 'C_in', 'C_out'],
)
@triton.jit
def conv1x1_bias_fused_kernel(
    weight_ptr, input_ptr, bias_ptr, output_ptr,
    NHW, C_in, HW, C_out,
    stride_in_n, stride_in_c,
    BLOCK_M: tl.constexpr,   # tile over C_out  (≥ 32 to cover 17 in one pass)
    BLOCK_N: tl.constexpr,   # tile over N*HW
    BLOCK_K: tl.constexpr,   # tile over C_in   (reduction)
):
    """
    output[nhw, m] = bias[m] + Σ_k  weight[m, k] * input[n, k, hw]
    where nhw = n * HW + hw  (flat index over batch × spatial)

    Shapes:
      weight  : [C_out, C_in]  (1×1 kernel dims folded)
      input   : [N, C_in, HW]  strides (stride_in_n, stride_in_c, 1)
      output  : [N, C_out, HW] contiguous  (= [NHW, C_out] when transposed)
    """
    pid_m   = tl.program_id(0)   # tile index over C_out
    pid_nhw = tl.program_id(1)   # tile index over N*HW

    offs_m   = pid_m   * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nhw = pid_nhw * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose flat nhw index → (batch, hw)
    n_idx  = offs_nhw // HW          # batch index for each nhw position
    hw_idx = offs_nhw - n_idx * HW   # spatial index   (= offs_nhw % HW)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, C_in, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # ---- weight tile [BLOCK_M, BLOCK_K] ----
        # weight[m, k] at weight_ptr + m * C_in + k
        w_ptrs = weight_ptr + offs_m[:, None] * C_in + offs_k[None, :]
        w_mask = (offs_m[:, None] < C_out) & (offs_k[None, :] < C_in)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # ---- input tile [BLOCK_K, BLOCK_N] ----
        # input[n, k, hw] = input_ptr + n*stride_in_n + k*stride_in_c + hw
        x_ptrs = (input_ptr
                  + n_idx[None, :]  * stride_in_n
                  + offs_k[:, None] * stride_in_c
                  + hw_idx[None, :])
        x_mask = (offs_k[:, None] < C_in) & (offs_nhw[None, :] < NHW)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w, x, allow_tf32=True)

    # Add bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < C_out, other=0.0)
    acc += bias[:, None]

    # ---- Store output ----
    # output layout: [N, C_out, HW]
    # output[n, m, hw] = output_ptr + n*C_out*HW + m*HW + hw
    out_ptrs = (output_ptr
                + n_idx[None, :]  * (C_out * HW)
                + offs_m[:, None] * HW
                + hw_idx[None, :])
    out_mask = (offs_m[:, None] < C_out) & (offs_nhw[None, :] < NHW)
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def conv1x1_bias_reshape_fused(in_0, in_1, in_2):
    """
    Fused replacement for:
        conv2d(in_2, in_1, in_0, stride=1, pad=0) * 1.0  →  reshape(-1, 17, 4096)

    in_0 : bias   [C_out=17]
    in_1 : weight [C_out=17, C_in=256, 1, 1]
    in_2 : input  [N, C_in=256, H=64, W=64]
    """
    N     = in_2.shape[0]
    C_in  = in_2.shape[1]
    H     = in_2.shape[2]
    W     = in_2.shape[3]
    C_out = in_1.shape[0]
    HW    = H * W      # 4096
    NHW   = N * HW     # total spatial tokens

    # Output: [N, C_out, HW] is already the desired (-1, C_out, HW) shape
    output = torch.empty((N, C_out, HW), dtype=in_2.dtype, device=in_2.device)

    # 2-D grid: (C_out tiles, N*HW tiles)
    grid = lambda META: (
        triton.cdiv(C_out, META['BLOCK_M']),
        triton.cdiv(NHW,   META['BLOCK_N']),
    )

    conv1x1_bias_fused_kernel[grid](
        in_1, in_2, in_0, output,
        NHW, C_in, HW, C_out,
        in_2.stride(0), in_2.stride(1),
    )

    return (output,)


def replacement_func():
    return conv1x1_bias_reshape_fused