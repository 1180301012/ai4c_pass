import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton GEMM kernel for 1x1 convolution
# Uses: C = W @ A
#   W [N=Cout, K=Cin]   row-major (weight)
#   A [K=Cin, M=B*H*W]  column-major viewed input
#   C [N=Cout, M=B*H*W] row-major output (= output[B, Cout, H, W] flattened)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    W_ptr,    # weight  [N, K]  row-major  (N=Cout, K=Cin)
    A_ptr,    # input   [K, M]  col-major  (K=Cin, M=B*H*W)
    bias_ptr, # bias    [N]
    C_ptr,    # output  [N, M]  row-major  (N=Cout, M=B*H*W)
    M, N, K,
    # W strides (row-major [N, K]):
    #   stride_wn = K = Cin,  stride_wk = 1
    stride_wn, stride_wk,
    # A strides (col-major [K, M]):
    #   stride_ak = M_per_batch = H*W,  stride_am = 1
    stride_ak, stride_am,
    # C strides (row-major [N, M]):
    #   stride_cn = M,  stride_cm = 1
    stride_cn, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BM] spatial indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BN] channel indices

    acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # [BK]

        # W tile [BN, BK]: W[n, k]
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # A tile [BK, BM]: A[k, m]  (stride_ak along K, stride_am=1 along M)
        a_ptrs = A_ptr + k_offs[:, None] * stride_ak + offs_m[None, :] * stride_am
        a_mask = (k_offs[:, None] < K) & (offs_m[None, :] < M)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # acc += W @ A  →  [BN, BK] @ [BK, BM] = [BN, BM]
        acc = tl.dot(w, a, acc)

    # Add bias [BN] broadcast to [BN, BM]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[:, None]

    # Store C[offs_n, offs_m]
    c_ptrs = C_ptr + offs_n[:, None] * stride_cn + offs_m[None, :] * stride_cm
    c_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


@torch.fx.wrap
def conv2d_1x1_triton(bias, weight, input_tensor):
    """
    Replaces torch.conv2d with Triton GEMM for 1x1 kernels.
    C = W @ A  where  W=weight[Cout,Cin], A=input[Cin,B*H*W] (col-major view).
    """
    B   = input_tensor.shape[0]
    Cin = input_tensor.shape[1]
    H   = input_tensor.shape[2]
    W   = input_tensor.shape[3]
    Cout = weight.shape[0]

    M = B * H * W   # total spatial positions
    N = Cout           # output channels
    K = Cin            # input channels

    output = torch.empty((B, Cout, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    # A [K=Cin, M=B*H*W] col-major viewed from input [B, Cin, H, W]
    #   stride_ak = H*W (step along Cin = stride_bh*stride_bw = W*1 = W... wait)
    #   Actually A[k, m] where m = b*HW + hw:
    #   input[b, k, hw] → offset = b*(Cin*HW) + k*HW + hw = k*HW + m (for same b)
    #   So stride_ak = HW, stride_am = 1
    stride_ak = H * W    # = HW (Cin stride in input tensor)
    stride_am = 1        # spatial positions are contiguous

    # W [N=Cout, K=Cin] row-major
    stride_wn = K        # = Cin
    stride_wk = 1

    # C [N=Cout, M=B*H*W] row-major
    stride_cn = M
    stride_cm = 1

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _conv1x1_gemm_kernel[grid](
        weight, input_tensor, bias, output,
        M, N, K,
        stride_wn, stride_wk,
        stride_ak, stride_am,
        stride_cn, stride_cm,
    )

    return output


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(bias, weight, input_tensor):
    """Match: torch.conv2d(input_tensor, weight, bias, (1,1), (0,0), (1,1), 1)"""
    return torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


def replacement_func():
    return conv2d_1x1_triton