"""
Mega-fused pass: captures BOTH independent chains in one pattern.

Chain 1: conv2d([1,2,1,8], [128,2,1,8]) -> view(1,2,8,8) -> sigmoid
         = linear([1,16] @ [128,16].T + bias) -> sigmoid -> [1,2,8,8]

Chain 2: in_3([1,2,8,8]).sum(dim=3,keepdim=True) -> in_3 / sum
         = row-wise L1 normalisation over last dim

Single Triton kernel with grid=(9,):
  programs 0-7: Chain 1 (each handles 16 outputs out of 128)
  program 8:    Chain 2 (handles all 16 rows x 8 cols at once)

Reduces 4 separate PyTorch kernel launches -> 1 Triton kernel launch.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern — captures the entire model graph
# ---------------------------------------------------------------------------

def pattern(x, weight, bias, inp3):
    c   = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    v   = c.view(1, 2, 8, 8)
    sig = v.sigmoid()
    s   = inp3.sum(dim=3, keepdim=True)
    norm = inp3 / s
    return (norm, sig)


def replacement_args(x, weight, bias, inp3):
    return (x, weight, bias, inp3)


# ---------------------------------------------------------------------------
# Mega-fused Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _mega_fuse_kernel(
    # Chain 1 pointers
    x_ptr, w_ptr, b_ptr, out_sig_ptr,
    # Chain 2 pointers
    inp3_ptr, out_norm_ptr,
    # Runtime scalars
    K,     # inner dim of linear = 16
    COLS,  # cols in inp3 = 8
    # Compile-time constants
    BLOCK_N: tl.constexpr,     # 16  — outputs per chain-1 program
    BLOCK_K: tl.constexpr,     # 16  — inner dim
    BLOCK_ROWS: tl.constexpr,  # 16  — rows for chain 2
    BLOCK_COLS: tl.constexpr,  # 8   — cols for chain 2
    N_CHAIN1: tl.constexpr,    # 8   — programs dedicated to chain 1
):
    pid = tl.program_id(0)

    if pid < N_CHAIN1:
        # ── Chain 1: linear + sigmoid ──────────────────────────────────────
        n_start = pid * BLOCK_N
        n_offs  = n_start + tl.arange(0, BLOCK_N)  # output indices
        k_offs  = tl.arange(0, BLOCK_K)             # inner dim indices

        # Load: input x [K], weight row-block [N, K], bias [N]
        x  = tl.load(x_ptr + k_offs)
        w  = tl.load(w_ptr + n_offs[:, None] * K + k_offs[None, :])
        b  = tl.load(b_ptr + n_offs)

        # Float32 accumulation for precision
        acc = tl.sum(x.to(tl.float32)[None, :] * w.to(tl.float32), axis=1) \
              + b.to(tl.float32)                       # [BLOCK_N]
        out = tl.sigmoid(acc).to(x.dtype)             # back to native dtype
        tl.store(out_sig_ptr + n_offs, out)

    else:
        # ── Chain 2: row-wise sum normalization ────────────────────────────
        r_offs = tl.arange(0, BLOCK_ROWS)             # [0..15]
        c_offs = tl.arange(0, BLOCK_COLS)             # [0..7]

        # Load entire [16, 8] block
        inp3   = tl.load(inp3_ptr + r_offs[:, None] * COLS + c_offs[None, :])

        # Compute row sums in float32
        inp3_f = inp3.to(tl.float32)
        s      = tl.sum(inp3_f, axis=1)               # [BLOCK_ROWS]

        # Normalise and store
        out = (inp3_f / s[:, None]).to(inp3.dtype)
        tl.store(out_norm_ptr + r_offs[:, None] * COLS + c_offs[None, :], out)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fuse_all(x, weight, bias, inp3):
    """
    x      : [1, 2, 1, 8]    — CUDA, native dtype
    weight : [128, 2, 1, 8]  — same dtype/device as x
    bias   : [128]            — same dtype/device as x
    inp3   : [1, 2, 8, 8]    — same dtype/device as x
    returns: (norm [1,2,8,8], sig [1,2,8,8])
    """
    device = x.device
    dtype  = x.dtype

    # Ensure everything is on the same device (no-op if already there)
    w  = weight.to(device=device)
    b  = bias.to(device=device)
    xc = x.contiguous()
    i3 = inp3.contiguous()

    # Allocate outputs
    out_sig  = torch.empty(128,          dtype=dtype, device=device)
    out_norm = torch.empty(1, 2, 8, 8,  dtype=dtype, device=device)

    # Single kernel launch: 8 programs for chain 1 + 1 for chain 2
    _mega_fuse_kernel[(9,)](
        xc, w, b, out_sig,
        i3, out_norm,
        16, 8,                   # K, COLS
        BLOCK_N=16,
        BLOCK_K=16,
        BLOCK_ROWS=16,
        BLOCK_COLS=8,
        N_CHAIN1=8,
    )

    # Return in model's return order: (norm, sig)
    return (out_norm, out_sig.view(1, 2, 8, 8))


def replacement_func():
    return fuse_all