import torch
import triton
import triton.language as tl


# =============================================================================
# Pattern: conv2d (1x1, stride=1, pad=0, dil=1, groups=1)
#          + dropout(p=0.0, train=False)   [no-op]
#          + add residual
# =============================================================================

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = tmp_3 + in_2
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# =============================================================================
# Triton GEMM kernel — TRANSPOSED formulation for better GPU occupancy
#
# Original:  C[M=C_out=128,  N=H*W=1024] = A[M=128,  K=256] @ B[K=256, N=1024]
#              → only 128/BM × 1024/BN blocks, BM≥16 ⟹ ≤8 M-tiles → poor occupancy
#
# Transposed: C[M=H*W=1024, N=C_out=128] = A[M=1024, K=256] @ B[K=256, N=128]
#              → 1024/BM × 128/BN blocks, BM=32 BN=16 ⟹ 32×8=256 blocks on 56 SMs ✓
#
# Memory layout (all NCHW with N_batch=1):
#   A = x  as [H*W, C_in]:   A[s, ic] = x_ptr + ic*H*W + s   → stride_am=1, stride_ak=H*W
#   B = weight as [C_in, C_out]: B[ic,oc] = w_ptr + oc*C_in + ic → stride_bk=1, stride_bn=C_in
#   C = output as [H*W, C_out]: C[s, oc] = out_ptr + oc*H*W + s → stride_cm=1, stride_cn=H*W
#   bias[C_out] broadcast over H*W
#   residual same layout as output
# =============================================================================

@triton.autotune(
    configs=[
        # M=1024, N=128, K=256
        # --- 512 blocks (9.1 waves on A30 56SMs): BLOCK_M=16 ---
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=6, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        # --- 256 blocks (4.6 waves): BLOCK_M=32 ---
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=6, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        # --- 128 blocks (2.3 waves): BLOCK_M=64 ---
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        # --- 64 blocks (1.1 waves): BLOCK_M=128 ---
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_conv1x1_add_kernel(
    # Pointers
    x_ptr, w_ptr, bias_ptr, res_ptr, out_ptr,
    # Dimensions (transposed): M=H*W, N=C_out, K=C_in
    M, N, K,
    # A=x strides in [H*W, C_in] view of NCHW:  stride_am=1, stride_ak=H*W
    stride_am, stride_ak,
    # B=weight strides in [C_in, C_out] view:   stride_bk=1, stride_bn=C_in
    stride_bk, stride_bn,
    # C/residual strides in [H*W, C_out] view:  stride_cm=1, stride_cn=H*W
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # L2-cache-friendly grouped mapping (group along M, the larger dim)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # channel-out offsets
    offs_k = tl.arange(0, BLOCK_K)                      # channel-in offsets

    mask_m = offs_m < M
    mask_n = offs_n < N

    # A = x[H*W, C_in]: contiguous in M (stride_am=1), strided in K (stride_ak=H*W)
    a_ptrs = x_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B = weight[C_in, C_out]: contiguous in K (stride_bk=1), strided in N (stride_bn=C_in)
    b_ptrs = w_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused: add bias (broadcast over M) + add residual
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :].to(tl.float32)

    mask_mn = mask_m[:, None] & mask_n[None, :]
    res_ptrs = res_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    res = tl.load(res_ptrs, mask=mask_mn, other=0.0)
    acc += res.to(tl.float32)

    out_ptrs = out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(out_ptrs, acc.to(res.dtype), mask=mask_mn)


@torch.fx.wrap
def fused_conv1x1_add(bias, weight, residual, x):
    """
    Fused 1×1 conv + dropout(p=0,train=False) + residual-add.

    in_0=bias:     [C_out]
    in_1=weight:   [C_out, C_in, 1, 1]
    in_2=residual: [N_batch, C_out, H, W]
    in_3=x:        [N_batch, C_in,  H, W]
    """
    N_batch, C_in, H, W = x.shape
    C_out = weight.shape[0]

    output = torch.empty((N_batch, C_out, H, W), device=x.device, dtype=x.dtype)

    # Transposed GEMM: M=H*W=1024, N=C_out=128, K=C_in=256
    M = N_batch * H * W   # 1024
    N = C_out              # 128
    K = C_in               # 256

    # A = x[H*W, C_in] in NCHW: A[s, ic] = x_ptr + ic*H*W + s
    stride_am = 1          # spatial stride
    stride_ak = H * W      # channel-in stride = 1024

    # B = weight[C_in, C_out]: weight[oc, ic] → B[ic, oc] = w_ptr + oc*C_in + ic
    stride_bk = 1          # ic stride (weight is stored [C_out, C_in], so ic stride = 1)
    stride_bn = C_in       # oc stride = 256

    # C/residual/output in [H*W, C_out] view of NCHW: C[s, oc] = ptr + oc*H*W + s
    stride_cm = 1          # spatial stride
    stride_cn = H * W      # channel-out stride = 1024

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    _fused_conv1x1_add_kernel[grid](
        x, weight, bias, residual, output,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    return output


def replacement_func():
    return fused_conv1x1_add