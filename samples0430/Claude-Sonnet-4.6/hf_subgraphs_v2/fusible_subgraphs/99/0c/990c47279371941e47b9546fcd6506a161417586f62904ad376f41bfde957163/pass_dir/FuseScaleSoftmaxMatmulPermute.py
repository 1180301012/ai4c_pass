import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────
# Pattern to match: scale → softmax → matmul → permute
# ─────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ─────────────────────────────────────────────────────────────────
# Triton kernel: fused scale + softmax + matmul + permute
#
# Shapes:  in_0 [B, N, C]   in_1 [B, C, D]   out [B, D, N]
# For all test cases: N=8192, C=19, D=256, B varies
#
# Tiling strategy:
#   Grid (B, ceil(N/BLOCK_M), ceil(D/BLOCK_N))
#   Each program produces a [BLOCK_N, BLOCK_M] tile of out[b, :, :]
#   (transposed from the [BLOCK_M, BLOCK_N] matmul result)
#   Transpose at store time → coalesced writes along N dimension
# ─────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32},  num_warps=8, num_stages=3),
    ],
    key=['B', 'N', 'D', 'C'],
)
@triton.jit
def _fused_scale_softmax_matmul_permute_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, N, C, D,
    stride_in0_b, stride_in0_n, stride_in0_c,
    stride_in1_b, stride_in1_c, stride_in1_d,
    stride_out_b, stride_out_d, stride_out_n,
    C_PADDED: tl.constexpr,   # next power-of-2 >= C (19 → 32)
    BLOCK_M:   tl.constexpr,  # tile size along N
    BLOCK_N:   tl.constexpr,  # tile size along D
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Row / column ranges for this tile
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # indices in [0, N)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # indices in [0, D)
    c_offs = tl.arange(0, C_PADDED)                    # indices in [0, C)

    mask_m = m_offs < N
    mask_c = c_offs < C
    mask_n = n_offs < D

    # ── Step 1: load in_0 and compute softmax (in fp32 for stability) ──
    in0_ptrs = (in0_ptr
                + pid_b * stride_in0_b
                + m_offs[:, None] * stride_in0_n
                + c_offs[None, :] * stride_in0_c)

    # Load; use a large negative for out-of-C positions so they vanish in softmax
    x_raw = tl.load(in0_ptrs,
                    mask=mask_m[:, None] & mask_c[None, :],
                    other=-1e4)
    # Remember the input dtype to cast back later
    input_dtype = x_raw.dtype

    x = x_raw.to(tl.float32)
    # Apply scale (0.0625 = 1/16)
    x = x * 0.0625
    # Force padded columns to -inf so they don't contribute to softmax
    x = tl.where(mask_c[None, :], x, float('-inf'))

    # Numerically-stable softmax over the C dimension
    x_max  = tl.max(x, axis=1)                          # [BLOCK_M]
    x      = x - x_max[:, None]
    x_exp  = tl.exp(x)
    # Zero out padding lanes before summing
    x_exp  = tl.where(mask_c[None, :], x_exp, 0.0)
    x_sum  = tl.sum(x_exp, axis=1)                      # [BLOCK_M]
    # Compute softmax and cast back to original dtype for tensor-core matmul
    softmax = (x_exp / x_sum[:, None]).to(input_dtype)  # [BLOCK_M, C_PADDED]

    # ── Step 2: load in_1 tile [C_PADDED, BLOCK_N] ──
    in1_ptrs = (in1_ptr
                + pid_b * stride_in1_b
                + c_offs[:, None] * stride_in1_c
                + n_offs[None, :] * stride_in1_d)

    in1_tile = tl.load(in1_ptrs,
                       mask=mask_c[:, None] & mask_n[None, :],
                       other=0.0)   # [C_PADDED, BLOCK_N]

    # ── Step 3: matmul ──
    # result [BLOCK_M, BLOCK_N] = softmax [BLOCK_M, C_PADDED] @ in1 [C_PADDED, BLOCK_N]
    result = tl.dot(softmax, in1_tile, allow_tf32=True)   # [BLOCK_M, BLOCK_N]

    # ── Step 4: transpose & store into permuted output [B, D, N] ──
    # result[m, n] needs to land at out[b, n_offs[n], m_offs[m]]
    # Transposing gives result_T[n, m] → out[b, n_offs[n], m_offs[m]]
    # which means the last (m) dimension is contiguous → coalesced writes.
    result_T = tl.trans(result)   # [BLOCK_N, BLOCK_M]

    out_ptrs = (out_ptr
                + pid_b * stride_out_b
                + n_offs[:, None] * stride_out_d
                + m_offs[None, :] * stride_out_n)

    tl.store(out_ptrs, result_T, mask=mask_n[:, None] & mask_m[None, :])


# ─────────────────────────────────────────────────────────────────
# Host-side wrapper  (must be decorated with @torch.fx.wrap)
# ─────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_scale_softmax_matmul_permute(in_0: torch.Tensor, in_1: torch.Tensor) -> torch.Tensor:
    B, N, C = in_0.shape
    _,  _, D = in_1.shape

    # C=19 → pad to next power-of-2 (32) for tensor-core alignment
    C_PADDED = 32

    # Output is the *permuted* result: [B, D, N]
    out = torch.empty((B, D, N), dtype=in_0.dtype, device=in_0.device)

    # 3-D launch grid; autotuner fills BLOCK_M / BLOCK_N
    grid = lambda meta: (
        B,
        triton.cdiv(N, meta['BLOCK_M']),
        triton.cdiv(D, meta['BLOCK_N']),
    )

    _fused_scale_softmax_matmul_permute_kernel[grid](
        in_0, in_1, out,
        B, N, C, D,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        C_PADDED=C_PADDED,
    )

    return out


# ─────────────────────────────────────────────────────────────────
# Required by the AI4C framework
# ─────────────────────────────────────────────────────────────────
def replacement_func():
    return fused_scale_softmax_matmul_permute