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
# Triton kernel: GELU + tiled coalesced transpose  [B,M,N] -> [B,N,M]
#
# Two-step strategy inside the wrapper:
#   1. PyTorch layer_norm  (already a highly optimised CUDA kernel)
#   2. This Triton kernel applies GELU and writes to the transposed layout
#      using small [TILE_M x TILE_N] tiles so that BOTH reads and writes
#      are coalesced (reads: stride-1 in N; writes: stride-1 in M via tl.trans)
#   Only ~8-16 float32 per thread -> no register spill, high occupancy.
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
    dst : [B, N, M]  contiguous (GELU applied + transposed)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    m_offs = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offs = pid_n * TILE_N + tl.arange(0, TILE_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Coalesced read [TILE_M, TILE_N] - stride-1 in N
    src_offs = pid_b * M * N + m_offs[:, None] * N + n_offs[None, :]
    x = tl.load(src_ptr + src_offs,
                 mask=m_mask[:, None] & n_mask[None, :],
                 other=0.0).to(tl.float32)

    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2 = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # Coalesced write [TILE_N, TILE_M] - stride-1 in M via tl.trans
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

    # Step 1: PyTorch's optimised layer_norm -> contiguous [B, M, N]
    ln_out = torch.nn.functional.layer_norm(in_2, (N,), in_1, in_0, 1e-5)

    # Step 2: Triton GELU + coalesced tiled transpose -> [B, N, M]
    out = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (triton.cdiv(M, meta['TILE_M']),
                triton.cdiv(N, meta['TILE_N']),
                B)

    _gelu_transpose_kernel[grid](ln_out, out, B, M, N)

    return out


# ---------------------------------------------------------------------------
# replacement_func - returns the callable (not the result of calling it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_layernorm_transpose_gelu