import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: GELU + tiled transpose  [B, M, N] → [B, N, M]
#
# Two-step strategy in the wrapper:
#   1. PyTorch layer_norm      (optimised CUDA kernel, coalesced I/O)
#   2. This Triton kernel      (GELU + coalesced tiled transpose)
#
# Memory access for this kernel:
#   Reads  [TILE_M, TILE_N] from contiguous src  → stride-1 in N → coalesced
#   Writes [TILE_N, TILE_M] to dst via tl.trans  → stride-1 in M → coalesced
#   Only 8–16 float32 per thread → no register spill
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 16, 'TILE_N': 16}, num_warps=2),
        triton.Config({'TILE_M': 16, 'TILE_N': 32}, num_warps=2),
        triton.Config({'TILE_M': 32, 'TILE_N': 32}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 64}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 32}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 64}, num_warps=8),
        triton.Config({'TILE_M': 128, 'TILE_N': 32}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 128}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def _gelu_transpose_kernel(
    src_ptr,
    dst_ptr,
    B,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """
    src : [B, M, N]  contiguous (layer_norm output)
    dst : [B, N, M]  contiguous (GELU applied, then transposed)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    m_offs = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offs = pid_n * TILE_N + tl.arange(0, TILE_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Coalesced read [TILE_M, TILE_N] – stride-1 in N
    src_offs = pid_b * M * N + m_offs[:, None] * N + n_offs[None, :]
    x = tl.load(src_ptr + src_offs,
                 mask=m_mask[:, None] & n_mask[None, :],
                 other=0.0).to(tl.float32)

    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2 = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # Coalesced write [TILE_N, TILE_M] – stride-1 in M via tl.trans
    dst_offs = pid_b * N * M + n_offs[:, None] * M + m_offs[None, :]
    tl.store(dst_ptr + dst_offs,
             tl.trans(y),
             mask=n_mask[:, None] & m_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_layernorm_transpose_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [N]         float16/bfloat16
    in_1 : weight [N]         float16/bfloat16
    in_2 : input  [B, M, N]   float16/bfloat16
    Returns [B, N, M]: layer_norm -> transpose(-2,-1) -> gelu
    """
    B, M, N = in_2.shape

    # Step 1: PyTorch's optimised layer_norm → contiguous [B, M, N]
    ln_out = torch.nn.functional.layer_norm(in_2, (N,), in_1, in_0, 1e-5)

    # Step 2: Triton GELU + coalesced tiled transpose → [B, N, M]
    out = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (triton.cdiv(M, meta['TILE_M']),
                triton.cdiv(N, meta['TILE_N']),
                B)

    _gelu_transpose_kernel[grid](ln_out, out, B, M, N)

    return out


# ---------------------------------------------------------------------------
# replacement_func – returns the callable (not the result of calling it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_layernorm_transpose_gelu
    configs=[
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps=8),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=8),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=16),
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps=8),
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps=16),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_ln_transpose_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    M,
    N,
    TILE_M:  tl.constexpr,   # rows per CTA
    BLOCK_N: tl.constexpr,   # >= N, power of 2
):
    """
    input  : [B, M, N]  row-major
    output : [B, N, M]  transposed
    One CTA handles TILE_M consecutive input rows.
    Reads are coalesced (stride-1 in N).
    Writes are coalesced (TILE_M consecutive M-positions per N value).
    """
    pid = tl.program_id(0)
    num_tiles_m = tl.cdiv(M, TILE_M)
    pid_b = pid // num_tiles_m
    pid_m = pid %  num_tiles_m

    m_base = pid_m * TILE_M
    m_offs = m_base + tl.arange(0, TILE_M)   # [TILE_M]
    n_offs = tl.arange(0, BLOCK_N)            # [BLOCK_N]

    m_mask = m_offs < M    # [TILE_M]
    n_mask = n_offs < N    # [BLOCK_N]

    # ------------------------------------------------------------------
    # 1. Load [TILE_M, BLOCK_N] block – coalesced reads (stride-1 in N)
    # ------------------------------------------------------------------
    in_offs = pid_b * M * N + m_offs[:, None] * N + n_offs[None, :]
    x = tl.load(input_ptr + in_offs,
                 mask=m_mask[:, None] & n_mask[None, :],
                 other=0.0).to(tl.float32)          # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 2. LayerNorm – reduce along axis=1 (N dimension)
    # ------------------------------------------------------------------
    mean  = tl.sum(x, axis=1) / N                   # [TILE_M]
    diff  = x - mean[:, None]                        # [TILE_M, BLOCK_N]
    var   = tl.sum(diff * diff, axis=1) / N          # [TILE_M]
    rstd  = 1.0 / tl.sqrt(var + 1e-5)               # [TILE_M]
    x_norm = diff * rstd[:, None]                    # [TILE_M, BLOCK_N]

    weight = tl.load(weight_ptr + n_offs,
                     mask=n_mask, other=1.0).to(tl.float32)   # [BLOCK_N]
    bias   = tl.load(bias_ptr   + n_offs,
                     mask=n_mask, other=0.0).to(tl.float32)   # [BLOCK_N]

    y = x_norm * weight[None, :] + bias[None, :]    # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 3. GELU (exact, erf): 0.5 * x * (1 + erf(x / sqrt(2)))
    # ------------------------------------------------------------------
    INV_SQRT2 = 0.7071067811865476
    y_gelu = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))  # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 4. Transpose + store → output[b, n, m]
    #    tl.trans gives [BLOCK_N, TILE_M]; offset n*M+m is stride-1 in m
    #    → TILE_M consecutive stores per N value → coalesced
    # ------------------------------------------------------------------
    out_offs = pid_b * N * M + n_offs[:, None] * M + m_offs[None, :]
    tl.store(output_ptr + out_offs,
             tl.trans(y_gelu),                       # [BLOCK_N, TILE_M]
             mask=n_mask[:, None] & m_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_layernorm_transpose_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [N]         float16 / bfloat16
    in_1 : weight [N]         float16 / bfloat16
    in_2 : input  [B, M, N]   float16 / bfloat16
    Returns: [B, N, M]  (layer_norm -> transpose(-2,-1) -> gelu)
    """
    B, M, N = in_2.shape
    out = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    # Grid depends on the autotuned TILE_M → use a lambda
    def grid(meta):
        return (triton.cdiv(B * M, meta['TILE_M']),)

    _fused_ln_transpose_gelu_kernel[grid](
        in_2,   # input_ptr
        in_1,   # weight_ptr
        in_0,   # bias_ptr
        out,
        B, M, N,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func – returns the callable (not the result of calling it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_layernorm_transpose_gelu