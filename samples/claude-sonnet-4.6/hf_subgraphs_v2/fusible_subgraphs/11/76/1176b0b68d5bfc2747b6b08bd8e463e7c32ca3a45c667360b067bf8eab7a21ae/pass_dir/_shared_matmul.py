"""
Shared Triton batched-GEMM kernel and dispatch wrapper.
Imported by both FuseBatchedMatmul.py and FuseBatchedMatmulAtOp.py so that
`replacement_func()` returns the exact same Python function object in both
passes (required to satisfy the replacement_func_limit constraint).
"""
import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel – generic tiled batched GEMM
#   A : [B, M, K]   (in_1 – typically row-major, stride_ak=1)
#   B : [B, K, N]   (in_0 – may be column-major / transposed, stride_bk=1)
#   C : [B, M, N]   output
#
#  B_TRANSPOSED=True  means B is stored column-major (stride_bk=1, stride_bn=K).
#  In this case we load BLOCK_N×BLOCK_K tiles and call tl.trans() to get a
#  coalesced [BLOCK_K×BLOCK_N] tile without copying the tensor first.
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # ── B_TRANSPOSED=True (yolo M=64, K=400, N=400) ──────────────────────
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        # ── B_TRANSPOSED=False (GCNet/S-ViPNAS N=1, various M/K) ─────────────
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        # Small-K fallback (e.g. S-ViPNAS K=48)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        # General / tiny-shape fallback
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'B_TRANSPOSED'],
)
@triton.jit
def _batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    B_TRANSPOSED: tl.constexpr,   # True when in_0 is stored column-major
):
    """
    C[b] = A[b] @ B[b]  for b in [0, B).
    Grid: (ceil(M/BLOCK_M)*ceil(N/BLOCK_N),  B).
    pid(0) encodes the flattened M×N tile; pid(1) = batch.
    All tiles for the same batch share the same A matrix → L2 reuse of A.
    """
    # Decode the flattened M×N tile index
    mn_tile  = tl.program_id(0)
    bid      = tl.program_id(1)

    N_tiles  = tl.cdiv(N, BLOCK_N)
    pid_m    = mn_tile // N_tiles
    pid_n    = mn_tile %  N_tiles

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer to the A-tile starting position
    a_ptrs = a_ptr + bid * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    if B_TRANSPOSED:
        # B stored column-major: element [k, n] lives at ptr + k*stride_bk + n*stride_bn
        # where stride_bk=1 and stride_bn=K_dim.
        # Load as [BLOCK_N, BLOCK_K] (rows along K → stride 1 → coalesced), then tl.trans().
        b_ptrs = b_ptr + bid * stride_bb + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    else:
        # B stored row-major: element [k, n] lives at ptr + k*stride_bk + n*stride_bn
        # where stride_bn=1 and stride_bk=N_dim.
        b_ptrs = b_ptr + bid * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem  = K - k * BLOCK_K
        k_mask = offs_k < k_rem

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        if B_TRANSPOSED:
            # Load [BLOCK_N, BLOCK_K], then transpose to [BLOCK_K, BLOCK_N]
            b_T = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            b   = tl.trans(b_T)
        else:
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk   # Advance K pointer (valid for both cases)

    c_ptrs = c_ptr + bid * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _run_kernel(in_0, in_1):
    """
    Compute  in_1 @ in_0  for 4-D inputs using the Triton kernel.
        in_1 : [B1, B2, M, K]   (A matrix – row-major assumed)
        in_0 : [B1, B2, K, N]   (B matrix – may be column-major / transposed)
    Returns [B1, B2, M, N].
    Avoids unnecessary .contiguous() copies by detecting and handling the
    transposed (column-major) B case inside the kernel with tl.trans().
    """
    # Make A (in_1) contiguous if needed (usually a no-op since v is row-major)
    a = in_1.contiguous() if in_1.stride(3) != 1 else in_1
    # Detect if B (in_0) is in column-major order (stride along last dim != 1)
    b_transposed = (in_0.stride(3) != 1)
    b = in_0  # Do NOT copy; handle layout in kernel

    B1, B2 = a.shape[0], a.shape[1]
    M,  K  = a.shape[2], a.shape[3]
    N      = b.shape[3]
    B      = B1 * B2

    out = torch.empty((B1, B2, M, N), dtype=in_1.dtype, device=in_1.device)

    stride_ab, stride_am, stride_ak = a.stride(1), a.stride(2), a.stride(3)
    stride_bb, stride_bk, stride_bn = b.stride(1), b.stride(2), b.stride(3)
    stride_cb, stride_cm, stride_cn = out.stride(1), out.stride(2), out.stride(3)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        B,
    )
    _batched_matmul_kernel[grid](
        a, b, out,
        B, M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        B_TRANSPOSED=b_transposed,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper – returned by BOTH pass files' replacement_func().
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def dispatch_matmul(in_0, in_1, route):
    if route == "torch_matmul":
        return _run_kernel(in_0, in_1)
    elif route == "at_op":
        return _run_kernel(in_0, in_1)
    else:
        return _run_kernel(in_0, in_1)