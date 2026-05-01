"""
Combined pass: fuse ALL operations in bat_resnext26ts start614_end618_2.

  in_1  →  sum(dim=2, keepdim=True)  →  in_1 / sum   →  tmp_1  [1,2,8,8]
  in_0  →  view(1,2,1,8,8)          →  expand(1,2,64,8,8)  →  tmp_3

Both independent sub-graphs are merged into ONE Triton kernel launch.

Grid design (1-D) — all programs use the SAME 1-D tensor width W:
  pids  0 .. BC-1          : normalise one (bc) slice of in_1
                              (2-pass loop over H rows; each step is a [W] vector)
  pids  BC .. BC+BC*E*H-1  : copy one row (bc,e,h, w:) from in_0 → out_expand
                              (single [W] load + store)

Using uniform [W]-shaped vectors in every branch avoids the 2D/1D
shape conflict that Triton's type-checker rejects when both paths are
compiled into the same PTX module.

With BC=2 and BC*E*H=1024, the grid has 1026 programs — filling all
56 SMs of an A30 in a single wave.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the ENTIRE forward function
# ---------------------------------------------------------------------------

def pattern(x, y):
    # --- normalise y (in_1) ---
    s    = y.sum(dim=2, keepdim=True)
    norm = y / s
    # --- view + expand x (in_0) ---
    v = x.view(1, 2, 1, 8, 8)
    e = v.expand(1, 2, 64, 8, 8)
    return (e, norm)


def replacement_args(x, y):
    return (x, y)


# ---------------------------------------------------------------------------
# Combined Triton kernel — UNIFORM [W] shapes in every branch
# ---------------------------------------------------------------------------

@triton.jit
def _combined_kernel(
    x_ptr,          # in_0  [1, 2, 8, 8]
    y_ptr,          # in_1  [1, 2, 8, 8]
    out0_ptr,       # expanded   [1, 2, 64, 8, 8]
    out1_ptr,       # normalised [1, 2, 8, 8]
    BC : tl.constexpr,   # B*C = 2
    H  : tl.constexpr,   # 8
    W  : tl.constexpr,   # 8
    E  : tl.constexpr,   # 64  (expand factor)
):
    pid = tl.program_id(0)
    w   = tl.arange(0, W)    # [W]  — used in BOTH branches (consistent type)

    if pid < BC:
        # ---- normalisation branch (pids 0..BC-1) --------------------------
        # 2-pass: accumulate sum per-column, then divide every row.
        bc   = pid
        base = bc * H * W

        # Pass 1 – accumulate fp32 column sums
        acc = tl.zeros([W], dtype=tl.float32)
        for h in range(H):
            row  = tl.load(y_ptr + base + h * W + w)
            acc += row.to(tl.float32)

        # Pass 2 – normalise and write
        for h in range(H):
            row     = tl.load(y_ptr + base + h * W + w)
            out_row = (row.to(tl.float32) / acc).to(row.dtype)
            tl.store(out1_ptr + base + h * W + w, out_row)

    else:
        # ---- expand-copy branch (pids BC..BC+BC*E*H-1) --------------------
        idx  = pid - BC              # in [0, BC*E*H)
        h    = idx % H
        idx2 = idx // H
        e    = idx2 % E
        bc   = idx2 // E

        # source  : x [bc, h, :]       at  bc*H*W + h*W + w
        src_base = bc * H * W + h * W
        # dest    : out0 [bc, e, h, :] at  bc*E*H*W + e*H*W + h*W + w
        dst_base = bc * E * H * W + e * H * W + h * W

        row = tl.load(x_ptr + src_base + w)
        tl.store(out0_ptr + dst_base + w, row)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def combined_sum_div_expand(x, y):
    """
    Single-kernel replacement for the full forward function:
      tmp_1 = in_1 / in_1.sum(dim=2, keepdim=True)   [1,2,8,8]
      tmp_3 = in_0.view(1,2,1,8,8).expand(1,2,64,8,8) [1,2,64,8,8]
    Returns (tmp_3, tmp_1).
    """
    B, C, H, W = 1, 2, 8, 8
    E  = 64
    BC = B * C   # = 2

    out0 = torch.empty((1, 2, 64, 8, 8), dtype=x.dtype, device=x.device)
    out1 = torch.empty_like(y)

    # Grid: 2 norm programs  +  BC*E*H = 1024 expand programs = 1026 total
    TOTAL = BC + BC * E * H

    _combined_kernel[(TOTAL,)](
        x, y, out0, out1,
        BC=BC, H=H, W=W, E=E,
        num_warps=1,
        num_stages=1,
    )

    return out0, out1


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return combined_sum_div_expand