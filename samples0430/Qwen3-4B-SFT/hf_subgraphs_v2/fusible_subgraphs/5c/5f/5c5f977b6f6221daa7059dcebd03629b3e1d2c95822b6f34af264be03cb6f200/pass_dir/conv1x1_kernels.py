"""
Shared Triton kernels for the fused conv1x1+stack+sum+cat pattern.
Imported by both Conv1x1SSCat_b2_c3.py and Conv1x1SSCat_b3_c2.py.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: 1x1 conv as GEMM (bias fused)
#
#   * a_ptr  : in_2 viewed as [N, K, M]  where K=Cin, M=H*W
#   * w_ptr  : in_1 viewed as [N, K]     weight rows × Cin cols
#   * b_ptr  : in_0 [N=Cout]              bias vector
#   * out_ptr: out  viewed as [N, M, N]   output [N, M, Cout]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, K, N,
    # A [M=N*H*W, K=Cin] → stride_am (m), stride_ak (cin)
    stride_am, stride_ak,
    # B [K=Cin, N=Cout] (weight read as transposed) → stride_bk (cin), stride_bn (cout)
    stride_bk, stride_bn,
    # C [M=N*H*W, N=Cout] → stride_cm (m), stride_cn (cout)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Standard GEMM: C[M,N] = A[M,K] @ B[K,N]
    A = input  [M, K]  (spatial M = N*H*W flattened; K = Cin, stride_k=H*W, stride_m=1)
    B = weight [K, N]  (weight[Cout,Cin] read transposed: stride_k=1 [Cin row], stride_n=Cin [Cout col])
    C = output [M, N]  (row=m stride=Cout, col=n stride=1)
    No tl.trans() — both tiles are in the correct logical [BK,BK]→[BK,BN] convention.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # A tile [BLOCK_M, BLOCK_K]: input at m-th spatial position, k-th channel
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # B tile [BLOCK_K, BLOCK_N]: weight transposed view
        #   b[k, n] = weight[n, k] = in_1_flat[n*Cin + k]
        #   k (cin) has stride 1; n (cout) has stride Cin
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # C += A @ B  →  [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, b, acc)

    # Bias fuse
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # Store [BLOCK_M, BLOCK_N]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(a_ptr.dtype.element_ty),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Kernel 2: Cat-half  (copy a channel-slice of cat_in into the second half of out)
#
#   cat_in : [N, Cout, H*W] contiguous
#   out    : [N, Cout, H*W] contiguous   (already has conv result in [:, :C1, :])
#
#   out[:, C1:, :, :] = cat_in  (i.e. out[n, C1+j, h, w] = cat_in[n, C1+j, h, w])
#
#   We view both as [N, M, C1+C2] and copy element j of each group m to the
#   appropriate lane in the destination's second half.
# ---------------------------------------------------------------------------
@triton.jit
def _cat_half_copy_kernel(
    cat_in_ptr,
    out_ptr,
    C_total,          # Cout = C1 + C2
    C1,               # channels from 1x1 conv
    C2,               # channels from other tensor
    M,                # H * W  (spatial positions per batch item)
    BLOCK_SIZE: tl.constexpr,   # elements per group handled per program (1024)
):
    """
    Scatter-copy: copy cat_in[:, C1:, :, :] into out[:, C1:, :, :].
    cat_in and out share the same [N, C_total, H*W] physical layout.
    One program per (n, m) row; copies C2 elements at flat offset n*C_total*M + m*C_total + C1.
    """
    pid = tl.program_id(0)   # 0 .. NN*M - 1

    n    = pid // M            # batch index
    m    = pid %  M            # spatial position within the image
    src_base = n * C_total * M + m * C_total   # flat start of (n, m) row in both tensors
    dst_base = src_base

    local = tl.arange(0, BLOCK_SIZE)
    mask  = local < C2

    val = tl.load(cat_in_ptr + src_base + C1 + local, mask=mask, other=0.)
    tl.store(out_ptr + dst_base + local, val, mask=mask)


# ---------------------------------------------------------------------------
# @torch.fx.wrap dispatcher — shared by both pass files
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_dispatch(in_0, in_1, in_2, in_3, route):
    """
    Fused: conv1x1(in_2, in_1, in_0) → stack([x],0) → sum(0) → cat([x, in_3], 1)
    Replaces the stacked-no-op + cat with a single GEMM + copy.

    Routes (ignored at runtime; kept for API compatibility):
      "route_a"  → concat(in_2 result, in_3)   [branch-in-2 | cat-in-3]
      "route_b"  → concat(in_3 result, in_2)   [branch-in-3 | cat-in-2]
    """
    if route == "route_a":
        bias, weight, branch, other = in_0, in_1, in_2, in_3
    else:
        bias, weight, branch, other = in_0, in_1, in_3, in_2

    N    = branch.shape[0]
    Cin  = branch.shape[1]
    H    = branch.shape[2]
    W    = branch.shape[3]
    Cout = bias.shape[0]
    M    = H * W

    # Allocate output [N, Cout, H, W]  =  cat([conv_result, other], dim=1)
    out = torch.empty((N, Cout, H, W), dtype=branch.dtype, device=branch.device)

    # -------------------------------------------------------------------
    # Step 1 – 1×1 conv as GEMM  (fill first C1 channels of out)
    # Tensors passed DIRECTLY as raw pointers; strides derived from shapes.
    # No .view()/.contiguous() calls (those emit blocked aten ops).
    # -------------------------------------------------------------------
    grid = lambda meta: (
        triton.cdiv(M,    meta['BLOCK_M']),
        triton.cdiv(Cout, meta['BLOCK_N']),
    )
    # a = branch [N, Cin, H, W]:  treats m = h*W + w as a flat dim
    #   cin stride = branch.stride(1) = H*W (= W here since layout is NCHW)
    #   m  stride = 1
    _conv1x1_gemm_kernel[grid](
        branch, weight, bias, out,
        M, Cin, Cout,
        # A = input [M=N*H*W, K=Cin]:
        #   m-stride = branch.stride(3) = 1  (consecutive spatial positions)
        #   k-stride = branch.stride(1) = H*W  (consecutive channels, NCHW)
        branch.stride(3), branch.stride(1),
        # B = weight [K=Cin, N=Cout] (transposed view):
        #   k-stride = weight.stride(1) = 1  (cin, inner dim of [Cout,Cin,1,1])
        #   n-stride = weight.stride(0) = Cin  (cout, outer dim)
        weight.stride(1), weight.stride(0),
        # C = output [M=N*H*W, N=Cout]:
        #   m-stride = out.stride(3) = 1
        #   n-stride = out.stride(1) = Cout
        out.stride(3), out.stride(1),
    )

    # -------------------------------------------------------------------
    # Step 2 – Copy other (C2 channels) into the second half of out.
    # Pass other directly; use other.stride() which returns int scalars.
    # -------------------------------------------------------------------
    C1 = Cout
    C2 = other.shape[1]
    NN = other.shape[0]
    M2 = M
    grid_copy = (NN * M2,)
    _cat_half_copy_kernel[grid_copy](
        other, out,
        C1 + C2,
        C1,
        C2,
        M2,
        BLOCK_SIZE=1024,
    )

    return (out,)