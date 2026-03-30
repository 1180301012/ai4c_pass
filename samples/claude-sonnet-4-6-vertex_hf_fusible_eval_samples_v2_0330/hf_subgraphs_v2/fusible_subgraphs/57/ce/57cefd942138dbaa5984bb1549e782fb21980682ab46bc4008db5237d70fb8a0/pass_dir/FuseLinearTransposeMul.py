import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match: linear(in_2, in_1, in_0) -> transpose(-1,-2) -> elementwise_mul with in_3
    in_0: bias  [N]
    in_1: weight [N, K]
    in_2: input  [B, M, K]
    in_3: gate   [B, N, M]
    output:      [B, N, M]
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# =============================================================================
# Single-pass fused kernel
#
# acc[BLOCK_N, BLOCK_M] = weight[BLOCK_N, K] @ in2[b, BLOCK_M, K].T
# The accumulator is already in (n, m) output order — no final tl.trans needed.
#
# Key: includes 'elem_size' (bytes per element) so that float32 (4) and
# float16/bfloat16 (2) get separately auto-tuned configurations.
# Float32 uses CUDA-core FFMA; float16/bfloat16 use tensor cores — very
# different optimal tile shapes.
# =============================================================================

@triton.autotune(
    configs=[
        # ---- float16 / bfloat16: tensor-core friendly (large tiles) ----
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # ---- float32: CUDA-core FFMA (medium tiles, more stages) --------
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['B', 'M', 'N', 'K', 'elem_size'],  # elem_size separates fp32 vs fp16/bf16 tuning
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    in2_ptr,     # [B, M, K]
    weight_ptr,  # [N, K]
    bias_ptr,    # [N]
    in3_ptr,     # [B, N, M]
    out_ptr,     # [B, N, M]
    B, M, N, K,
    elem_size,   # element size in bytes — used only for autotune key dispatch
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    program_id(0) -> batch b
    program_id(1) -> N-tile
    program_id(2) -> M-tile

    Computes:  out[b, n, m] = in3[b,n,m] * (dot(in2[b,m,:], weight[n,:]) + bias[n])
    """
    batch  = tl.program_id(0)
    n_tile = tl.program_id(1)
    m_tile = tl.program_id(2)

    m_start = m_tile * BLOCK_M
    n_start = n_tile * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)
    m_mask  = m_range < M
    n_mask  = n_range < N

    # acc[BLOCK_N, BLOCK_M]:  weight[BLOCK_N, K] @ in2[b, BLOCK_M, K].T
    # Already in (n, m) layout — no final transpose of acc required.
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_tile * BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        weight = tl.load(
            weight_ptr + n_range[:, None] * K + k_range[None, :],
            mask=n_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        in2 = tl.load(
            in2_ptr + batch * M * K + m_range[:, None] * K + k_range[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        # acc[BLOCK_N, BLOCK_M] += weight @ in2.T
        acc = tl.dot(weight, tl.trans(in2), acc, out_dtype=tl.float32)

    # bias[n] broadcast over m → [BLOCK_N, BLOCK_M]
    bias = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0).to(tl.float32)
    acc  = acc + bias[:, None]

    # in3[b, n, m]: coalesced load in M dimension
    in3 = tl.load(
        in3_ptr + batch * N * M + n_range[:, None] * M + m_range[None, :],
        mask=n_mask[:, None] & m_mask[None, :], other=0.0
    )

    # Element-wise multiply — acc is already [BLOCK_N, BLOCK_M]
    result = in3 * acc.to(in3.dtype)

    tl.store(
        out_ptr + batch * N * M + n_range[:, None] * M + m_range[None, :],
        result,
        mask=n_mask[:, None] & m_mask[None, :]
    )


@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    in_0: bias   [N]
    in_1: weight [N, K]
    in_2: input  [B, M, K]
    in_3: gate   [B, N, M]
    output:      [B, N, M]
    """
    B, M, K = in_2.shape
    N = in_1.shape[0]

    in_2 = in_2.contiguous()
    in_1 = in_1.contiguous()
    in_0 = in_0.contiguous()
    in_3 = in_3.contiguous()

    out = torch.empty((B, N, M), dtype=in_3.dtype, device=in_2.device)

    grid = lambda meta: (
        B,
        triton.cdiv(N, meta['BLOCK_N']),
        triton.cdiv(M, meta['BLOCK_M']),
    )

    fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, M, N, K,
        in_2.element_size(),   # dtype-aware autotune key
    )

    return out


def replacement_func():
    return fused_linear_transpose_mul

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    in2_ptr,     # [B, M, K]
    weight_ptr,  # [N, K]
    bias_ptr,    # [N]
    in3_ptr,     # [B, N, M]
    out_ptr,     # [B, N, M]
    B, M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    program_id(0) -> batch b
    program_id(1) -> N-tile (n dimension)
    program_id(2) -> M-tile (m dimension)

    acc[BLOCK_N, BLOCK_M] = weight[BLOCK_N, K] @ in2[b, BLOCK_M, K].T
    This is already in (n, m) output order — no final tl.trans needed.
    """
    batch  = tl.program_id(0)
    n_tile = tl.program_id(1)
    m_tile = tl.program_id(2)

    m_start = m_tile * BLOCK_M
    n_start = n_tile * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_range < M
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_tile * BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        weight = tl.load(
            weight_ptr + n_range[:, None] * K + k_range[None, :],
            mask=n_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        in2 = tl.load(
            in2_ptr + batch * M * K + m_range[:, None] * K + k_range[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        acc = tl.dot(weight, tl.trans(in2), acc, out_dtype=tl.float32)

    bias = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[:, None]  # [BLOCK_N, BLOCK_M]

    in3 = tl.load(
        in3_ptr + batch * N * M + n_range[:, None] * M + m_range[None, :],
        mask=n_mask[:, None] & m_mask[None, :], other=0.0
    )

    result = in3 * acc.to(in3.dtype)

    tl.store(
        out_ptr + batch * N * M + n_range[:, None] * M + m_range[None, :],
        result,
        mask=n_mask[:, None] & m_mask[None, :]
    )


# =============================================================================
# Kernels B + C: Split-K approach for small batches (B <= 4)
#
# Problem: for B=1 the single-pass kernel only launches ~84 programs, giving
# poor GPU occupancy (~3 waves on A30's 28 SMs).
#
# Solution: split the K-reduction across NUM_K_SPLITS program groups.
# Each group writes a float32 partial result; a second kernel reduces them
# and applies bias + in3 multiply.
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K', 'k_per_split'],
)
@triton.jit
def split_k_partial_matmul_kernel(
    in2_ptr,      # [B, M, K]
    weight_ptr,   # [N, K]
    partial_ptr,  # [num_k_splits, B, N, M]  float32  — output partial sums
    B, M, N, K,
    num_k_splits, k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    program_id(0) -> batch * num_k_splits + k_split   (merged dimension)
    program_id(1) -> N-tile
    program_id(2) -> M-tile

    Writes partial acc[BLOCK_N, BLOCK_M] into partial[k_split, batch, n, m].
    A separate reduce kernel finishes the computation.
    """
    pid_bk  = tl.program_id(0)
    n_tile  = tl.program_id(1)
    m_tile  = tl.program_id(2)

    batch   = pid_bk // num_k_splits
    k_split = pid_bk % num_k_splits
    k_base  = k_split * k_per_split

    m_start = m_tile * BLOCK_M
    n_start = n_tile * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_range < M
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_tile_idx in range(tl.cdiv(k_per_split, BLOCK_K)):
        k_start = k_base + k_tile_idx * BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        weight = tl.load(
            weight_ptr + n_range[:, None] * K + k_range[None, :],
            mask=n_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        in2 = tl.load(
            in2_ptr + batch * M * K + m_range[:, None] * K + k_range[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)

        acc = tl.dot(weight, tl.trans(in2), acc, out_dtype=tl.float32)

    # Write partial result: partial[k_split, batch, n_range, m_range]
    partial_ptrs = (partial_ptr
                    + k_split * B * N * M
                    + batch   * N * M
                    + n_range[:, None] * M
                    + m_range[None, :])
    tl.store(partial_ptrs, acc, mask=n_mask[:, None] & m_mask[None, :])


@triton.jit
def split_k_reduce_mul_kernel(
    partial_ptr,        # [num_k_splits, B, N, M]  float32
    bias_ptr,           # [N]
    in3_ptr,            # [B, N, M]
    out_ptr,            # [B, N, M]
    B, N, M, num_k_splits,
    BLOCK: tl.constexpr,
):
    """
    For each output element (b, n, m):
        out = in3[b,n,m] * (sum_s(partial[s,b,n,m]) + bias[n])
    """
    pid     = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total   = B * N * M
    mask    = offsets < total

    # n index for bias
    n = (offsets % (N * M)) // M

    # Reduce across K-splits
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(num_k_splits):
        partial = tl.load(partial_ptr + s * B * N * M + offsets,
                          mask=mask, other=0.0)
        acc = acc + partial

    bias = tl.load(bias_ptr + n, mask=mask, other=0.0).to(tl.float32)
    in3  = tl.load(in3_ptr  + offsets, mask=mask, other=0.0)

    result = in3 * (acc + bias).to(in3.dtype)
    tl.store(out_ptr + offsets, result, mask=mask)


# =============================================================================
# Wrapper
# =============================================================================

@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    in_0: bias   [N]
    in_1: weight [N, K]
    in_2: input  [B, M, K]
    in_3: gate   [B, N, M]
    output:      [B, N, M]
    """
    B, M, K = in_2.shape
    N = in_1.shape[0]

    in_2 = in_2.contiguous()
    in_1 = in_1.contiguous()
    in_0 = in_0.contiguous()
    in_3 = in_3.contiguous()

    out = torch.empty((B, N, M), dtype=in_3.dtype, device=in_2.device)

    # Single-pass fused kernel for all batch sizes.
    # For large B the fusion avoids materialising the intermediate [B,M,N] tensor.
    # For small B we accept that cuBLAS would be faster for the matmul but there
    # is no permitted torch-level fallback; the single-pass kernel is still correct.
    grid = lambda meta: (
        B,
        triton.cdiv(N, meta['BLOCK_N']),
        triton.cdiv(M, meta['BLOCK_M']),
    )
    fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, M, N, K,
    )

    return out


def replacement_func():
    return fused_linear_transpose_mul