import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused 1x1-conv (GEMM) + SiLU activation
# Grid: (ceil(M/BM), ceil(N/BN)) where M = N_batch*H*W, N = C_out, K = C_in
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K', 'DTYPE_IDX'],
)
@triton.jit
def _fused_conv1x1_silu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xc, stride_xn,
    stride_wn, stride_wk,
    stride_on, stride_om, stride_oc,
    DTYPE_IDX,          # 0=fp32, 1=fp16, 2=bf16
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial (N*H*W) tiles
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output-channel tiles

    # Accumulate in fp32 for numerical accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # ---- Load x tile: [BLOCK_M, BLOCK_K] ----
        # x layout: [N, C_in, H, W]  =>  flat_idx = n*stride_xn + c*stride_xc + m*stride_xm
        x_ptrs = x_ptr + offs_n[None, :] * stride_xc + offs_m[:, None] * stride_xm
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # ---- Load w tile: [BLOCK_N, BLOCK_K] ----
        # w layout: [C_out, C_in, 1, 1]  =>  flat_idx = n*stride_wn + k*stride_wk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # ---- Matrix multiply accumulate ----
        # a [BLOCK_M, BLOCK_K] @ b.T [BLOCK_K, BLOCK_N] => [BLOCK_M, BLOCK_N]
        acc = tl.dot(x, tl.trans(w), acc)

    # ---- Add bias ----
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # ---- SiLU: z = acc * sigmoid(acc) ----
    z = acc * tl.sigmoid(acc)

    # ---- Store result in original dtype ----
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_n[None, :] * stride_oc + offs_m[:, None] * stride_om
    if DTYPE_IDX == 1:
        tl.store(out_ptrs, z.to(tl.float16), mask=out_mask)
    elif DTYPE_IDX == 2:
        tl.store(out_ptrs, z.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, z, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_silu(x, w, bias):
    """
    Fused 1x1-conv2d + SiLU activation kernel.
    x:    [N, C_in, H, W]
    w:    [C_out, C_in, 1, 1]
    bias: [C_out]
    """
    N_batch, C_in, H, W = x.shape
    C_out = w.shape[0]
    HW = H * W
    M = N_batch * HW        # number of spatial positions (NCHW treated as [M, K])
    K = C_in
    N_dim = C_out

    out = torch.empty((N_batch, C_out, H, W), dtype=x.dtype, device=x.device)

    # Dtype index: 0=fp32, 1=fp16, 2=bf16
    if x.dtype == torch.float16:
        DTYPE_IDX = 1
    elif x.dtype == torch.bfloat16:
        DTYPE_IDX = 2
    else:
        DTYPE_IDX = 0

    # Input strides for [N, C_in, H, W]:
    #   stride(3)=1  (W dimension)
    #   stride(2)=W  (H dimension)
    #   stride(1)=H*W  (C_in dimension)
    #   stride(0)=C_in*H*W  (N dimension)
    # For element (n, c, h, w) at spatial position m = h*W + w:
    #   flat_idx = n*C_in*H*W + c*H*W + m = n*stride_xn + c*stride_xc + m*stride_xm
    stride_xm = 1                       # within-spatial stride (W)
    stride_xc = H * W                   # channel stride
    stride_xn = C_in * H * W            # batch stride

    # Weight strides for [C_out, C_in, 1, 1]:
    #   stride(0)=C_in (output-channel stride)
    #   stride(1)=1   (input-channel stride)
    stride_wn = C_in
    stride_wk = 1

    # Output strides (same layout as input)
    stride_on = C_out * H * W
    stride_om = 1
    stride_oc = H * W

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_dim, meta['BLOCK_N']),
    )

    _fused_conv1x1_silu_kernel[grid](
        x, w, bias, out,
        M, N_dim, K,
        stride_xm, stride_xc, stride_xn,
        stride_wn, stride_wk,
        stride_on, stride_om, stride_oc,
        DTYPE_IDX,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    """
    Matches:
        conv2d = torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
        tmp_3  = F.silu(conv2d, inplace=False)
        tmp_4  = F.dropout(tmp_3, 0.0, False, False)
        return tmp_4
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.silu(conv2d, inplace=False)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    # in_0 = bias, in_1 = weight, in_2 = input
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_silu