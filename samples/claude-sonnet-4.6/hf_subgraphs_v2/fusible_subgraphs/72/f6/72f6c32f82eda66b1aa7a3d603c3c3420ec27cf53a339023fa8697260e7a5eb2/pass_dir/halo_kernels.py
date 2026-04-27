"""
Shared Triton kernels for halo-attention layout fusion.
Imported by HaloLayoutFusion80.py and HaloLayoutFusion48.py.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton GEMM kernel  (for 1×1 conv as matrix multiply)
# A [M, K] × B [K, N] → C [M, N]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_k(
    A, B, C,
    M, N, K,
    sAm, sAk,
    sBk, sBn,
    sCm, sCn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_n
    group_id = pid // num_in_group
    first_m  = group_id * GROUP_M
    group_sz = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % num_in_group) % group_sz
    pid_n = (pid % num_in_group) // group_sz

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptrs = A + rm[:, None] * sAm + rk[None, :] * sAk
    B_ptrs = B + rk[:, None] * sBk + rn[None, :] * sBn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs,
                    mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_K),
                    other=0.0)
        b = tl.load(B_ptrs,
                    mask=(rk[:, None] < K - k * BLOCK_K) & (rn[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        A_ptrs += BLOCK_K * sAk
        B_ptrs += BLOCK_K * sBk

    C_ptrs = C + rm[:, None] * sCm + rn[None, :] * sCn
    tl.store(C_ptrs, acc.to(C.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


# ---------------------------------------------------------------------------
# Unified layout scatter kernel
# Reads conv_out [C_out, H*W], writes out_k and out_v
# All geometry is constexpr so the compiler eliminates all divisions
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _layout_k(
    cin_ptr, ok_ptr, ov_ptr,
    N,
    H:           tl.constexpr,
    W:           tl.constexpr,
    C_per_head:  tl.constexpr,
    split1:      tl.constexpr,
    split2:      tl.constexpr,
    n_wins:      tl.constexpr,
    n_wins_w:    tl.constexpr,
    win_size:    tl.constexpr,
    stride_s:    tl.constexpr,
    pad_s:       tl.constexpr,
    n_pix:       tl.constexpr,
    BLOCK_SIZE:  tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Decode: offs = head * (n_wins*n_pix*C_per_head) + win*(n_pix*C_per_head) + pix*C_per_head + d
    d    = offs % C_per_head;    tmp = offs // C_per_head
    pix  = tmp % n_pix;          tmp = tmp // n_pix
    win  = tmp % n_wins;         head = tmp // n_wins

    fh = pix // win_size;  fw = pix % win_size
    i  = win // n_wins_w;   j  = win  % n_wins_w
    h  = i * stride_s + fh - pad_s
    w  = j * stride_s + fw - pad_s
    valid = mask & (h >= 0) & (h < H) & (w >= 0) & (w < W)

    c   = head * C_per_head + d
    val = tl.load(cin_ptr + c * (H * W) + h * W + w, mask=valid, other=0.0)

    is_k = d < split1

    # out_k [n_heads, n_wins, split1, n_pix]
    ki = head * (n_wins * split1 * n_pix) + win * (split1 * n_pix) + d * n_pix + pix
    tl.store(ok_ptr + ki, val, mask=mask & is_k)

    # out_v [n_heads, n_wins, n_pix, split2]
    sv = tl.where(is_k, 0, d - split1)
    vi = head * (n_wins * n_pix * split2) + win * (n_pix * split2) + pix * split2 + sv
    tl.store(ov_ptr + vi, val, mask=mask & ~is_k)


# ---------------------------------------------------------------------------
# Route-specific Python helpers
# ---------------------------------------------------------------------------
def _run_halo_80(in_0_gpu, in_1):
    """
    in_0_gpu: weight  [640, 512, 1, 1] on CUDA
    in_1    : input   [1,  512, 16, 16] on CUDA
    Returns : (out_k [8,4,16,144], out_v [8,4,144,64])
    """
    C_out, C_in, H, W = 640, 512, 16, 16
    N = H * W                             # 256
    dtype  = in_1.dtype
    device = in_1.device

    A = in_0_gpu.view(C_out, C_in)        # [640, 512]
    B = in_1.view(C_in, N)                # [512, 256]
    conv_out = torch.empty(C_out, N, dtype=dtype, device=device)

    grid_g = lambda META: (triton.cdiv(C_out, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _matmul_k[grid_g](
        A, B, conv_out,
        C_out, N, C_in,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        conv_out.stride(0), conv_out.stride(1),
    )

    n_heads, n_wins, n_pix = 8, 4, 144
    out_k = torch.empty(n_heads, n_wins, 16, n_pix, dtype=dtype, device=device)
    out_v = torch.empty(n_heads, n_wins, n_pix, 64, dtype=dtype, device=device)

    N_lay = n_heads * n_wins * n_pix * 80        # 368640
    grid_l = lambda META: (triton.cdiv(N_lay, META['BLOCK_SIZE']),)
    _layout_k[grid_l](
        conv_out, out_k, out_v, N_lay,
        H=H, W=W,
        C_per_head=80, split1=16, split2=64,
        n_wins=4, n_wins_w=2, win_size=12, stride_s=8, pad_s=2, n_pix=144,
    )
    return (out_k, out_v)


def _run_halo_48(in_0_gpu, in_1):
    """
    in_0_gpu: weight  [384, 256, 1, 1] on CUDA
    in_1    : input   [1,  256, 16, 16] on CUDA
    Returns : (out_k [8,4,16,144], out_v [8,4,144,32])
    """
    C_out, C_in, H, W = 384, 256, 16, 16
    N = H * W                             # 256
    dtype  = in_1.dtype
    device = in_1.device

    A = in_0_gpu.view(C_out, C_in)        # [384, 256]
    B = in_1.view(C_in, N)                # [256, 256]
    conv_out = torch.empty(C_out, N, dtype=dtype, device=device)

    grid_g = lambda META: (triton.cdiv(C_out, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _matmul_k[grid_g](
        A, B, conv_out,
        C_out, N, C_in,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        conv_out.stride(0), conv_out.stride(1),
    )

    n_heads, n_wins, n_pix = 8, 4, 144
    out_k = torch.empty(n_heads, n_wins, 16, n_pix, dtype=dtype, device=device)
    out_v = torch.empty(n_heads, n_wins, n_pix, 32, dtype=dtype, device=device)

    N_lay = n_heads * n_wins * n_pix * 48        # 221184
    grid_l = lambda META: (triton.cdiv(N_lay, META['BLOCK_SIZE']),)
    _layout_k[grid_l](
        conv_out, out_k, out_v, N_lay,
        H=H, W=W,
        C_per_head=48, split1=16, split2=32,
        n_wins=4, n_wins_w=2, win_size=12, stride_s=8, pad_s=2, n_pix=144,
    )
    return (out_k, out_v)


# ---------------------------------------------------------------------------
# Shared @torch.fx.wrap dispatch – SAME function object in BOTH pass files
# ---------------------------------------------------------------------------
@torch.fx.wrap
def halo_fused_dispatch(in_0, in_1, route):
    """
    in_0  : weight tensor (may be on CPU – will be moved to in_1.device)
    in_1  : activation tensor on CUDA
    route : "r80" | "r48"
    """
    in_0_gpu = torch.as_tensor(in_0, device=in_1.device)
    if route == "r80":
        return _run_halo_80(in_0_gpu, in_1)
    elif route == "r48":
        return _run_halo_48(in_0_gpu, in_1)