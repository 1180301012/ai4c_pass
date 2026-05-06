"""
Shared Triton kernels for the two optimization patterns used across all pass files.
All pass files import these and define a common dispatch_wrapper.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: 1x1 Conv2d + View  [B, Cin, H, W] + weight[Cout, Cin] + bias[Cout]
#           → [B, Cout, H*W]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 256, 'BLOCK_K': 64,  'BLOCK_S': 64,  'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_K': 128, 'BLOCK_S': 64,  'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_K': 64,  'BLOCK_S': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_K': 128, 'BLOCK_S': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_C': 64,  'BLOCK_K': 64,  'BLOCK_S': 128, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_K': 64,  'BLOCK_S': 64,  'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_K': 64,  'BLOCK_S': 128, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_C': 64,  'BLOCK_K': 128, 'BLOCK_S': 128, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_K': 256, 'BLOCK_S': 32,  'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_K': 512, 'BLOCK_S': 16,  'num_warps': 8, 'num_stages': 1}),
    ],
    key=['B', 'Cout', 'Cin', 'S'],
)
@triton.jit
def _conv1x1_view_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, Cout, Cin, S,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    1x1 conv as batched matmul.
    Input [B, Cin, S]  (NCHW spatially collapsed), weight [Cout, Cin].
    Load B=[BLOCK_C,BLOCK_K] (coalesced in K), A^T=[BLOCK_K,BLOCK_S] (coalesced in S).
    acc[BLOCK_C, BLOCK_S] = B @ A^T + bias
    Output [B, Cout, S] with stride [Cout*S, S, 1].
    Grid: (B, ceil(Cout/BLOCK_C))
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_start = pid_c * BLOCK_C
    c_offs  = c_start + tl.arange(0, BLOCK_C)
    c_mask  = c_offs < Cout

    acc = tl.zeros((BLOCK_C, BLOCK_S), dtype=tl.float32)

    base_w = weight_ptr
    base_b = input_ptr + pid_b * Cin * S

    for k_start in range(0, Cin, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Cin

        # Weight tile B: [BLOCK_C, BLOCK_K] — K inner (stride-1) → coalesced ✓
        b_ptrs = base_w + c_offs[:, None] * Cin + k_offs[None, :]
        b_mask = c_mask[:, None] & k_mask[None, :]
        # Load as fp16/bf16 tile: shape [BLOCK_C, BLOCK_K]
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Input tile A: [BLOCK_K, BLOCK_S] — S inner (stride-1) → coalesced ✓
        a_ptrs = base_b + k_offs[:, None] * S + tl.arange(0, BLOCK_S)[None, :]
        a_mask = k_mask[:, None] & tl.arange(0, BLOCK_S)[None, :] < S
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # a_tile: [BLOCK_K, BLOCK_S], b_tile: [BLOCK_C, BLOCK_K]
        # tl.dot(b_tile, a_tile) = [BLOCK_C, BLOCK_K] @ [BLOCK_K, BLOCK_S] = [BLOCK_C, BLOCK_S]
        acc = acc + tl.dot(b_tile, a_tile, out_dtype=tl.float32)

    # Add bias
    bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
    acc  = acc + bias[:, None]

    out_base = output_ptr + pid_b * Cout * S
    out_ptrs = out_base + c_offs[:, None] * S + tl.arange(0, BLOCK_S)[None, :]
    out_mask = c_mask[:, None] & tl.arange(0, BLOCK_S)[None, :] < S
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def _run_conv_view(bias, weight, x):
    """Implementation: runs the fused conv1x1 + view kernel."""
    B    = x.shape[0]
    Cin  = x.shape[1]
    H    = x.shape[2]
    W    = x.shape[3]
    Cout = weight.shape[0]
    S    = H * W

    out  = torch.empty((B, Cout, S), dtype=x.dtype, device=x.device)
    grid = lambda meta: (B, triton.cdiv(Cout, meta['BLOCK_C']))

    _conv1x1_view_kernel[grid](x, weight, bias, out, B, Cout, Cin, S)
    return out


# ---------------------------------------------------------------------------
# Kernel 2: Reduction  x[B, S, C]  →  out[B, 1, C]   (mean over dim=-2=dim S)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 128, 'num_warps': 8}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 256, 'num_warps': 8}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 64,  'num_warps': 4}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 128, 'num_warps': 4}),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 256, 'num_warps': 4}),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 32,  'num_warps': 2}),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 64,  'num_warps': 2}),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 128, 'num_warps': 4}),
    ],
    key=['C', 'S'],
)
@triton.jit
def _mean_dim1_kernel(
    x_ptr, out_ptr,
    B, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Reduces x[B, S, C] along axis 1 (dim S) to produce out[B, 1, C].
    acc[BLOCK_C] = sum over S dimension for channels c_offs
    Each program handles one batch × BLOCK_C channel tile.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_start = pid_c * BLOCK_C
    c_offs  = c_start + tl.arange(0, BLOCK_C)
    c_mask  = c_offs < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S

        # Load x[pid_b, s_offs, c_offs]: shape [BLOCK_S, BLOCK_C]
        x_ptrs = x_ptr + pid_b * (S * C) + s_offs[:, None] * C + c_offs[None, :]
        x_mask = s_mask[:, None] & c_mask[None, :]
        tile   = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc = acc + tl.sum(tile.to(tl.float32), axis=0)

    out_vals = acc / S   # 1D [BLOCK_C] (float32 mean over S)
    out_ptrs = out_ptr + pid_b * C + c_offs   # 1D [BLOCK_C]
    tl.store(out_ptrs, out_vals.to(x_ptr.dtype.element_ty), mask=c_mask)


def _run_mean_dim1(y):
    """Implementation: Triton reduction over dim=-2 (dim 1 in [B,S,C])."""
    B, S, C = y.shape
    out  = torch.empty((B, 1, C), dtype=y.dtype, device=y.device)
    grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))
    _mean_dim1_kernel[grid](y, out, B, C, S)
    return out


# ---------------------------------------------------------------------------
# Shared dispatch wrapper — ALL pass files return THIS function from
# replacement_func().  Identical code across all files.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_wrapper(*args):
    """
    Routes:
      "conv_view"  → fused 1x1-conv + view
      "mean_dim1"  → reduction mean over dim=-2 (dim 1) with keepdim=True
    The last positional arg (args[-1]) is the route string.
    """
    route = args[-1]
    if route == "conv_view":
        return _run_conv_view(args[0], args[1], args[2])
    elif route == "mean_dim1":
        return _run_mean_dim1(args[0])
    # Fallback (should never be reached)
    return args[0]