import torch
from pass_dir.shared_dispatch import _shared_dispatch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_silu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_wm, stride_wk,
    stride_xk, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1 conv + SiLU kernel.

    Computes:  output[oc, n_hw] = SiLU( sum_ic weight[oc,ic] * input[n_hw,ic] + bias[oc] )

    A = weight  [M, K]  (oc x ic),  stride_wm=K, stride_wk=1
    B = input   [K, N]  (ic x n_hw),stride_xk=H*W, stride_xn=1
    C = output  [M, N]  (oc x n_hw), stride_om=N,  stride_on=1
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offs = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- GEMM loop over K (= Cin = 128) ----
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load weight tile A[BM, BK]:  weight[oc, ic]
        # addr = oc * stride_wm + ic * stride_wk  (ic=K index, coalesced along ic)
        a_ptrs = weight_ptr + m_offs[:, None] * stride_wm + k_offs[None, :] * stride_wk
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(a_ptrs, mask=a_mask, other=0.0)   # [BM, BK]

        # Load input tile B[BK, BN]:  input[n, ic, h, w] → treated as [ic, n_hw]
        # addr = ic * stride_xk + n_hw * stride_xn
        # For contiguous NCHW [1,C,H,W]: ic stride = H*W, n_hw stride = 1
        b_ptrs = input_ptr + k_offs[:, None] * stride_xk + n_offs[None, :] * stride_xn
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        w = tl.load(b_ptrs, mask=b_mask, other=0.0)   # [BK, BN]

        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # ---- Add bias [M] ----
    bias = tl.load(bias_ptr + m_offs, mask=m_offs < M, other=0.0)
    acc = acc + bias[:, None].to(tl.float32)

    # ---- Apply SiLU: x * sigmoid(x) ----
    acc_silu = acc * tl.sigmoid(acc).to(tl.float32)

    # ---- Store output [M, N] ----
    out_ptrs = output_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(out_ptrs, acc_silu.to(output_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_silu(bias, weight, x):
    """
    Fused 1x1 Conv2d + SiLU + NoDropout(p=0.0) implementation.

    Args:
        bias:   [Cout]            — in_0
        weight: [Cout, Cin, 1, 1] — in_1
        x:      [N, Cin, H, W]    — in_2

    Returns:
        out:    [N, Cout, H, W]
    """
    N_batch = x.shape[0]
    Cin     = x.shape[1]
    H       = x.shape[2]
    W       = x.shape[3]
    Cout    = weight.shape[0]
    HW      = H * W
    M       = Cout           # OC tiles: [M]
    K       = Cin            # IC tiles: [K] = [128]
    N_hw    = N_batch * HW   # spatial positions: [16]

    stride_wm = weight.stride(0)   # stride along Cout dim = Cin = K (=128)
    stride_wk = weight.stride(1)   # stride along Cin dim   = 1
    stride_xk = HW                  # stride along Cin for NCHW = H*W
    stride_xn = 1
    stride_om = N_hw                # stride along Cout for NCHW = N_batch*H*W
    stride_on = 1

    out = torch.empty((N_batch, Cout, H, W), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N_hw, META['BLOCK_N']),
    )

    _conv1x1_silu_kernel[grid](
        x, weight, bias, out,
        M, N_hw, K,
        stride_wm, stride_wk,
        stride_xk, stride_xn,
        stride_om, stride_on,
    )
    return out


# ──────────────────────────────────────────────────────────────────
# Pattern / replacement interface used by the AI4C pass framework
# ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    """
    Test torch.conv2d as a single Python-level anchor (no silu/dropout).
    This MATCHES because the target graph IS at the Python-level torch.conv2d.
    """
    return torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1, in_2):
    # Pass (bias, weight, input) to the fused conv1x1+silu kernel
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_silu