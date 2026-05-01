"""
Conv1x1Bias.py
Replace torch.conv2d(x, w, b, (1,1), (0,0), (1,1), 1) – a 1×1 convolution –
with a Triton batched GEMM kernel.

Formulation:
  out[n, c_out, hw] = sum_k  w[c_out, k] * x[n, k, hw]  + b[c_out]

Grid: (cdiv(C_out,BM) * cdiv(HW,BN), N)
  GROUP_SIZE_M=8 grouping maximises L2 reuse of x tiles.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # small-tile, high-occupancy  (good for all N, fp32-safe)
        triton.Config({'BM':  64, 'BN':  64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM':  64, 'BN':  64, 'BK': 32}, num_warps=4, num_stages=4),
        triton.Config({'BM':  64, 'BN':  64, 'BK': 32}, num_warps=4, num_stages=5),  # ← fp32: 80KB ✓
        triton.Config({'BM':  64, 'BN':  64, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM':  64, 'BN':  64, 'BK': 64}, num_warps=4, num_stages=4),
        # medium-tile
        triton.Config({'BM': 128, 'BN':  64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN':  64, 'BK': 32}, num_warps=4, num_stages=4),
        triton.Config({'BM':  64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM':  64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN':  64, 'BK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM':  64, 'BN': 128, 'BK': 64}, num_warps=4, num_stages=3),
        # large-tile balanced  (better arithmetic intensity)
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=4),
        # wide HW tile – fewer blocks, higher L2 reuse
        triton.Config({'BM':  64, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM':  64, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=4),
        triton.Config({'BM':  64, 'BN': 256, 'BK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BM':  64, 'BN': 256, 'BK': 64}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_warps=8, num_stages=3),
        # tall C_out tile
        triton.Config({'BM': 256, 'BN':  64, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN':  64, 'BK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
        # very-wide BN=512 — near single-wave for small N (e.g. N=2→64 blocks)
        triton.Config({'BM':  64, 'BN': 512, 'BK': 32}, num_warps=8, num_stages=2),  # 72KB fp16 ✓
        triton.Config({'BM': 128, 'BN': 512, 'BK': 32}, num_warps=8, num_stages=2),  # 80KB fp16 ✓
    ],
    key=['N', 'C_in', 'C_out', 'HW'],
)
@triton.jit
def _conv1x1_bias_kernel(
    x_ptr,    # [N, C_in, H, W]   contiguous (NCHW)
    w_ptr,    # [C_out, C_in, 1, 1] contiguous → treat as [C_out, C_in]
    b_ptr,    # [C_out]
    out_ptr,  # [N, C_out, H, W]  contiguous
    N, C_in, C_out, HW,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """
    Grid axis-0: (pid_m, pid_n) tiles with GROUP_SIZE_M=8 L2 grouping.
    Grid axis-1: batch index n.
    """
    pid = tl.program_id(0)
    n   = tl.program_id(1)

    num_pid_m = tl.cdiv(C_out, BM)
    num_pid_n = tl.cdiv(HW,    BN)

    GROUP_SIZE_M: tl.constexpr = 8
    num_in_group = GROUP_SIZE_M * num_pid_n
    group_id     = pid // num_in_group
    first_pid_m  = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % num_in_group) % group_size_m
    pid_n = (pid % num_in_group) // group_size_m

    m_offs = pid_m * BM + tl.arange(0, BM)
    n_offs = pid_n * BN + tl.arange(0, BN)
    m_mask = m_offs < C_out
    n_mask = n_offs < HW

    acc = tl.zeros([BM, BN], dtype=tl.float32)

    for k_start in range(0, C_in, BK):
        k_offs = k_start + tl.arange(0, BK)
        k_mask = k_offs < C_in

        w_ptrs = w_ptr + m_offs[:, None] * C_in + k_offs[None, :]
        w_tile = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        x_ptrs = x_ptr + n * C_in * HW + k_offs[:, None] * HW + n_offs[None, :]
        x_tile = tl.load(x_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = acc + tl.dot(w_tile, x_tile, out_dtype=tl.float32, allow_tf32=True)

    b_tile = tl.load(b_ptr + m_offs, mask=m_mask, other=0.0).to(tl.float32)
    acc    = acc + b_tile[:, None]

    out_ptrs = out_ptr + n * C_out * HW + m_offs[:, None] * HW + n_offs[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@torch.fx.wrap
def triton_conv1x1_bias(x, w, b):
    """
    Drop-in replacement for torch.conv2d(x, w, b, (1,1), (0,0), (1,1), 1).
    x:   [N, C_in, H, W]
    w:   [C_out, C_in, 1, 1]
    b:   [C_out]
    out: [N, C_out, H, W]  same dtype & device as x
    """
    N     = x.shape[0]
    C_in  = x.shape[1]
    H     = x.shape[2]
    W     = x.shape[3]
    C_out = w.shape[0]
    HW    = H * W

    out = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(C_out, meta['BM']) * triton.cdiv(HW, meta['BN']),
        N,
    )

    _conv1x1_bias_kernel[grid](x, w, b, out, N, C_in, C_out, HW)
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API expected by AI4C framework
# ---------------------------------------------------------------------------

def pattern(x, w, b):
    return torch.conv2d(x, w, b, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(x, w, b):
    return (x, w, b)


def replacement_func():
    return triton_conv1x1_bias