import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        # Small BLOCK_M → more blocks → better SM occupancy for small M (batch=1)
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        # Medium BLOCK_M
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        # wider N tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'C_out', 'C_in'],
)
@triton.jit
def fused_conv1x1_hardtanh_mul_kernel(
    in2_ptr,       # [N, C_in, H, W]   NCHW layout
    weight_ptr,    # [C_out, C_in]  (from [C_out, C_in, 1, 1])
    bias_ptr,      # [C_out]
    in3_ptr,       # [N, C_out, H, W]  NCHW layout
    out_ptr,       # [N, C_out, H, W]  NCHW layout
    C_in, C_out, HW, M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1 Conv + hardtanh(0,6) + element-wise multiply.

    Grid: (ceil(M/BLOCK_M), ceil(C_out/BLOCK_N))

    Proven correct GEMM tiling:
      weight  [BLOCK_K, BLOCK_N] — N fast → coalesced
      in2     [BLOCK_M, BLOCK_K] — K fast → coalesced
      acc[M,N] += in2[M,K] × weight[K,N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] flat spatial
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] output channels

    # Decompose flat spatial index into (batch, hw)
    n_idx  = offs_m // HW   # [BLOCK_M]
    hw_idx = offs_m % HW    # [BLOCK_M]

    # Accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load in2[BLOCK_M, BLOCK_K] — K is last dim → coalesced
        in2_offsets = (n_idx[:, None] * C_in * HW
                       + offs_k[None, :] * HW
                       + hw_idx[:, None])   # [BLOCK_M, BLOCK_K]
        in2_mask    = (offs_m[:, None] < M) & (offs_k[None, :] < C_in)
        in2         = tl.load(in2_ptr + in2_offsets,
                               mask=in2_mask, other=0.0).to(tl.float32)

        # Load weight[BLOCK_K, BLOCK_N] — N is last dim → coalesced
        w_offsets = offs_k[:, None] + offs_n[None, :] * C_in  # [BLOCK_K, BLOCK_N]
        w_mask    = (offs_k[:, None] < C_in) & (offs_n[None, :] < C_out)
        w         = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

        # [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc = tl.dot(in2, w, acc)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < C_out, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # hardtanh / clamp to [0, 6]
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    # Load in3[BLOCK_M, BLOCK_N] and multiply
    in3_offsets = (n_idx[:, None] * C_out * HW
                   + offs_n[None, :] * HW
                   + hw_idx[:, None])           # [BLOCK_M, BLOCK_N]
    in3_mask    = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    in3         = tl.load(in3_ptr + in3_offsets,
                           mask=in3_mask, other=0.0).to(tl.float32)
    acc = acc * in3

    # Store — N is last dim → coalesced
    tl.store(out_ptr + in3_offsets, acc.to(in3_ptr.dtype.element_ty), mask=in3_mask)


@torch.fx.wrap
def fused_conv1x1_hardtanh_mul(in_0, in_1, in_2, in_3):
    """
    Replacement for:  conv2d(in_2, in_1, in_0, stride=1, pad=0, dil=1, groups=1)
                      * hardtanh(in_3, 0, 6)
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : input  [N, C_in, H, W]
    in_3 : activation [N, C_out, H, W]
    Returns: [N, C_out, H, W]
    """
    N, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]
    HW    = H * W
    M     = N * HW

    out = torch.empty_like(in_3)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(C_out, meta['BLOCK_N']))

    fused_conv1x1_hardtanh_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        C_in, C_out, HW, M,
    )
    return out


def replacement_func():
    return fused_conv1x1_hardtanh_mul