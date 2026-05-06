import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Full fusion: 1x1-conv(viewed as GEMM) + hardtanh(relu6) + element-wise mul
#
# The pattern matches the ENTIRE computation graph:
#   conv2d(in_2, in_1, in_0, stride=1) * hardtanh(in_3)
#
# This avoids materialising the large conv2d output tensor (~288 MB for
# batch=128, ~6 MB for batch=1) and reduces memory traffic from ~1.28 GB
# to ~716 MB – a ~1.8x bandwidth saving.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Triton GEMM kernel for 1x1 convolution
# Tensors are NCHW with H and W as the "inner" spatial dims.
#
# We treat each (b, h, w) position as a "row" m = b*H*W + h*W + w.
# The GEMM is:
#   A[m, k]  = input[b, k, h, w]   (stride [IC*HW, HW, 1])
#   B[k', n] = weight[n, k]        (stride [IC, 1], treated as [OC, IC])
#   C[m, n]  = GEMM(A, B) + bias   (stride [1, HW])
# Then apply relu6 and multiply by the second input tensor.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 512,  'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512,  'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 2048, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 2048, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=16, num_stages=2),
    ],
    key=['M', 'IC', 'OC'],
)
@triton.jit
def conv1x1_gemm_kernel(
    a_ptr,       # input  : [B, IC, H, W] contiguous NCHW
    b_ptr,       # weight : [OC, IC] (squeezed 1x1 dims)
    bias_ptr,    # bias   : [OC]
    c2_ptr,      # second tensor (hardtanh input): [B, OC, H, W]
    out_ptr,     # output : [B, OC, H, W]
    M,           # B * H * W  (rows of the GEMM)
    N,           # OC         (cols of the GEMM)
    K,           # IC         (K for GEMM)
    stride_am,   # A row stride = IC * HW
    stride_ak,   # A col stride = HW
    stride_bk,   # B row stride = IC
    stride_bn,   # B col stride = 1  (the "K" dim of B is OC)
    stride_cm,   # C row stride = OC * HW
    HW,          # H * W
    OC,          # output channels (for index decomposition)
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)

    # 2-D tile indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_OC

    offs_m = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]  flattened batch+spatial
    offs_n = n_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    # Decompose flat GEMM row index into batch and spatial
    # offs_m = b * HW + hw  (with b >= 0, hw in [0, HW))
    b_idx   = offs_m // HW              # [BLOCK_M]
    hw_idx  = offs_m % HW               # [BLOCK_M]
    out_nhw = b_idx * (OC * HW) + hw_idx   # flat index in [OC, HW] layout

    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)

    # ------------------------------------------------------------------
    # GEMM loop over the input channels K (IC)
    # ------------------------------------------------------------------
    for k in range(0, K, BLOCK_IC):
        offs_k = k + tl.arange(0, BLOCK_IC)

        # --- Load A: input[b, k_off, hw] = a_ptr[b*IC*HW + k*HW + hw]
        #     A has shape [BLOCK_M, BLOCK_IC] with strides [IC*HW, HW]
        a_ptrs = (a_ptr
                  + b_idx[:, None] * (K * HW)
                  + offs_k[None, :] * HW
                  + hw_idx[:, None])
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)   # [BLOCK_M, BLOCK_IC]

        # --- Load B: weight[n, k] = b_ptr[n*IC + k]
        #     B has shape [BLOCK_OC, BLOCK_IC] with strides [IC, 1]
        b_ptrs = (b_ptr
                  + offs_n[:, None] * stride_bk
                  + offs_k[None, :] * stride_bn)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)   # [BLOCK_OC, BLOCK_IC]

        # matmul and accumulate  [BLOCK_M, BLOCK_IC] @ [BLOCK_IC, BLOCK_OC]
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

    # Bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # ReLU(6) = clamp to [0, 6]
    acc = tl.minimum(tl.maximum(acc, 0.0), 6.0)

    # Load the hardtanh input (second tensor) and compute final multiply
    c2_ptrs = out_nhw[:, None] + offs_n[None, :] * HW
    c2_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c2_data = tl.load(c2_ptr + c2_ptrs, mask=c2_mask, other=0.0)

    result = acc * c2_data

    # Store output
    out_ptrs = (out_ptr
                + b_idx[:, None] * (N * HW)
                + offs_n[None, :] * HW
                + hw_idx[:, None])
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty), mask=c2_mask)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv1x1_hardtanh_mul(in_0, in_1, in_2, in_3):
    """
    Fused 1x1-conv (GEMM) + hardtanh + element-wise multiply.

    in_0 : bias   [OC]
    in_1 : weight [OC, IC, 1, 1]
    in_2 : input  [B,  IC, H, W]
    in_3 : second tensor [B, OC, H, W]
    """
    B   = in_2.shape[0]
    IC  = in_2.shape[1]
    H   = in_2.shape[2]
    W   = in_2.shape[3]
    OC  = in_1.shape[0]
    HW  = H * W
    M   = B * HW
    N   = OC
    K   = IC

    out = torch.empty_like(in_3)

    grid = lambda meta: (
        triton.cdiv(M,       meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_OC']),
    )

    conv1x1_gemm_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        M, N, K,
        IC * HW,   # stride_am  (A: batch stride)
        HW,        # stride_ak  (A: channel stride)
        IC,        # stride_bk  (B: output-channel stride)
        1,         # stride_bn  (B: input-channel stride)
        OC * HW,   # stride_cm  (C2/out: output-channel stride)
        HW,        # HW
        OC,        # OC
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_conv1x1_hardtanh_mul