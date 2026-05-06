import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: 1×1 conv  →  ×1.0  →  reshape(-1, 17, 4096)
# Shapes used:
#   bias   : [17]
#   weight : [17, 256, 1, 1]
#   input  : [B, 256, 64, 64]  (B ∈ {1,4,32,64,512})
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel
#
# Grid: 2-D  (ceil(M / BLOCK_M),  ceil(N / BLOCK_N))
#   where M = B * HW  (flattened batch × spatial)
#         N = N_out   = 17
#         K = C_in    = 256
#
# For each tile (pid_m, pid_n):
#   • Load input  tile: [BLOCK_M, BLOCK_K]  (strided)
#   • Load weight tile: [BLOCK_N, BLOCK_K]  (contiguous rows)
#   • acc += A_tile @ B_tile^T              ( → [BLOCK_M, BLOCK_N] )
#   • Add bias (broadcast over M dimension)
#   • Store   [BLOCK_M, BLOCK_N] tile to output
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # BLOCK_M × BLOCK_N × BLOCK_K triplets (all power-of-2)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 1024,'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        # wider BLOCK_K
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['B', 'C_in', 'HW'],
)
@triton.jit
def _fused_conv1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B, C_in, HW, N,
    stride_ib, stride_ic, stride_im,
    stride_wn, stride_wk,
    stride_ob, stride_on, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row / column ranges for this tile
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2-D index into the virtual [B*HW, C_in] matrix
    b_idx = m_offs // HW     # which batch element
    hw_idx = m_offs % HW     # which spatial position

    # Pointer base for inputs of batch b (advance by b * C_in * HW)
    A_base = input_ptr + b_idx[:, None] * (C_in * HW)

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(C_in, BLOCK_K)):
        k_off = k * BLOCK_K
        k_range = k_off + tl.arange(0, BLOCK_K)

        # ── Load A tile: [BLOCK_M, BLOCK_K] ──────────────────────────────
        # a[m, k] = input[b_idx[m], k_range[k], hw_idx[m]]
        a_ptrs = A_base + k_range[None, :] * HW + hw_idx[:, None]
        a_mask = (m_offs[:, None] < B * HW) & (k_range[None, :] < C_in)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # dtype = input dtype

        # ── Load B tile: [BLOCK_N, BLOCK_K] ──────────────────────────────
        # b[n, k] = weight[n_offs[n], k_range[k]]
        #   weight strides: [C_in, 1, HW, HW] → stride(0)=C_in, stride(1)=1
        b_ptrs = weight_ptr + n_offs[:, None] * stride_wn + k_range[None, :] * stride_wk
        b_mask = (n_offs[:, None] < N) & (k_range[None, :] < C_in)
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # dtype = weight dtype

        # ── Accumulate: [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N] ─────────
        acc += tl.dot(A_tile, tl.trans(B_tile), out_dtype=tl.float32)

    # ── Bias (broadcast over M) ──────────────────────────────────────────
    bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # ── Store output ─────────────────────────────────────────────────────
    m_mask = m_offs < B * HW
    out_ptrs = (output_ptr
                + b_idx[:, None] * stride_ob
                + n_offs[None, :] * stride_on
                + hw_idx[:, None] * stride_om)
    out_mask = m_mask[:, None] & (n_offs[None, :] < N)
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper (must be @torch.fx.wrap so the graph rewriter can call it)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_conv1x1_mul1_reshape(in_0, in_1, in_2):
    """
    in_0 : bias   [N]           = [17]
    in_1 : weight [N, C_in, 1, 1] = [17, 256, 1, 1]
    in_2 : input  [B, C_in, H, W]
    Returns: [B, N, H*W]
    """
    B    = in_2.shape[0]
    C_in = in_2.shape[1]
    H    = in_2.shape[2]
    W    = in_2.shape[3]
    HW   = H * W
    N    = in_0.shape[0]

    output = torch.empty((B, N, HW), dtype=in_2.dtype, device=in_2.device)

    # Grid covers (M_tiles, N_tiles)
    grid = lambda meta: (
        triton.cdiv(B * HW, meta['BLOCK_M']),
        triton.cdiv(N,      meta['BLOCK_N']),
    )

    _fused_conv1x1_kernel[grid](
        in_2,      # input
        in_1,      # weight
        in_0,      # bias
        output,
        B, C_in, HW, N,
        in_2.stride(0), in_2.stride(1), in_2.stride(3),
        in_1.stride(0), in_1.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
    )

    return output


def replacement_func():
    return fused_conv1x1_mul1_reshape