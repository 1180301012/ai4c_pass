"""
Pass: FuseFullForward
Matches the ENTIRE model forward computation and replaces it with a SINGLE Triton
kernel launch (vs 4 native PyTorch kernel launches in baseline).

Grid: (128,) — one program per output element
Each program computes:
  • GEMV+sigmoid:   sigmoid_out[prog] = sigmoid(W[prog,:] @ x + b[prog])
  • Row-norm:       div_out[prog]     = in3[prog] / sum(in3[row,:])
    where row = prog // 8, reusing the same 16-wide SIMD (mask for 8 active elements)

Pattern returns: (tmp_6, tmp_4) matching model's return tuple exactly.

Design:
  - _run_combined_forward (@torch.fx.wrap) is OPAQUE to FX → 1 opaque node
  - fuse_full_forward (NOT wrapped) is TRACED by FX → result[0]/result[1] give
    2 separate FX getitem nodes → 2 returning nodes matching the pattern's 2
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  — must mirror model.py EXACTLY including return tuple order
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return (tmp_6, tmp_4)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Single combined Triton kernel
#
# Grid: (128,)  — one program per output index (= output channel for GEMV
#                  = flat element index for row-norm)
#
# SIMD width: C_IN = 16  (used for GEMV)
# Row-norm reuses the same 16-wide arange with mask to select only N_COLS=8
# ---------------------------------------------------------------------------
@triton.jit
def combined_forward_kernel(
    in2_ptr,          # [C_IN=16] flat input (1,2,1,8 contiguous)
    in1_ptr,          # [C_OUT=128, C_IN=16] flat weight (128,2,1,8 contiguous)
    in0_ptr,          # [C_OUT=128] bias
    in3_ptr,          # [128] flat data (1,2,8,8 contiguous)
    sigmoid_out_ptr,  # [C_OUT=128] output for sigmoid path
    div_out_ptr,      # [128]       output for row-norm path
    C_IN:   tl.constexpr,   # = 16
    N_COLS: tl.constexpr,   # = 8
):
    prog    = tl.program_id(0)          # 0 .. 127
    in_offs = tl.arange(0, C_IN)        # [0,1,...,15]

    # ── GEMV + bias + sigmoid ──────────────────────────────────────────────
    x   = tl.load(in2_ptr + in_offs).to(tl.float32)              # [16]
    w   = tl.load(in1_ptr + prog * C_IN + in_offs).to(tl.float32)# [16]
    dot = tl.sum(x * w, axis=0)
    b   = tl.load(in0_ptr + prog).to(tl.float32)
    tl.store(sigmoid_out_ptr + prog, tl.sigmoid(dot + b))

    # ── Row-wise L1 normalisation ──────────────────────────────────────────
    # prog = row * N_COLS + col  →  row = prog // N_COLS
    row      = prog // N_COLS
    row_offs = tl.arange(0, N_COLS)                                    # [8], no masking needed
    row_data = tl.load(in3_ptr + row * N_COLS + row_offs).to(tl.float32)  # [8] clean load
    row_sum  = tl.sum(row_data, axis=0)
    # Load specific element (cache-hot after row_data load)
    elem = tl.load(in3_ptr + prog).to(tl.float32)
    tl.store(div_out_ptr + prog, elem / row_sum)

# ---------------------------------------------------------------------------
# Wrapper — opaque to FX (returns a Python tuple proxy)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _run_combined_forward(in_0, in_1, in_2, in_3):
    """
    Launches ONE Triton kernel that computes both outputs simultaneously.
    Returns (sigmoid_out [1,2,8,8], div_out [1,2,8,8]).
    """
    sigmoid_flat = torch.empty(128, dtype=in_2.dtype, device=in_2.device)
    div_out      = torch.empty_like(in_3)

    combined_forward_kernel[(128,)](
        in_2, in_1, in_0, in_3,
        sigmoid_flat, div_out,
        C_IN=16, N_COLS=8,
    )
    return (sigmoid_flat.view(1, 2, 8, 8), div_out)


# ---------------------------------------------------------------------------
# Top-level replacement — NOT wrapped, so FX traces through it.
# result[0] and result[1] become two separate getitem nodes in the FX graph,
# giving exactly 2 returning nodes to match the pattern's 2 returning nodes.
# ---------------------------------------------------------------------------
def fuse_full_forward(in_0, in_1, in_2, in_3):
    result      = _run_combined_forward(in_0, in_1, in_2, in_3)
    sigmoid_out = result[0]   # FX getitem node  →  tmp_4
    div_out     = result[1]   # FX getitem node  →  tmp_6
    return (div_out, sigmoid_out)   # must match model return order (tmp_6, tmp_4)


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return fuse_full_forward