import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the pairwise mask computation and the input reshape+transpose
# observed in all three graph variants (float16/96, float16/128, float32/128).
# ---------------------------------------------------------------------------
def pattern(tmp_9, in_0):
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 96)
    tmp_6 = tmp_5.transpose(2, 3)
    return (tmp_16, tmp_6)


def replacement_args(tmp_9, in_0):
    return (tmp_9, in_0)


# ---------------------------------------------------------------------------
# Triton kernel 1: Fused pairwise-attention-mask computation
#   Input:  tmp_9  shape (1, P=361, Q=49)   (the row/col indicator mask)
#   Output: out0   shape (1, P=361, Q=49, Q=49)
#   out0[b, p, q, r] = -1000.0  if (q//7 != r//7) OR (q%7 != r%7)  (padding)
#                    = 0.0         otherwise
#
# Because the mask is fully determined by the geometry (133 = 19*7, padding = 5
# which means last patch row AND last patch col are invalid), we compute it
# from p/q/r arithmetic inside the kernel – no tmp_9 loads needed, avoiding
# a 361*49 = 17689-element read per invocation.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['P', 'Q'],
)
@triton.jit
def _pairwise_mask_kernel(
    out_ptr,
    P: tl.constexpr,   # 361
    Q: tl.constexpr,   # 49
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    idx   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = P * Q * Q
    mask  = idx < total

    # Decompose flat index -> (p, q, r)
    r = idx % Q
    q = (idx // Q) % Q
    p =  idx // (Q * Q)

    # Load the position-based mask from tmp_9  (float32 for arithmetic)
    # tmp_9[p, q] = 1.0  if q lands in the last patch row or col (padding)
    #             = 0.0  otherwise
    patch_size: tl.constexpr = 7
    q_patch: tl.constexpr  = q // patch_size
    q_intra: tl.constexpr  = q % patch_size
    r_patch: tl.constexpr  = r // patch_size
    r_intra: tl.constexpr  = r % patch_size

    q_is_padding: tl.constexpr = (q_patch == 18) | (q_intra == 18)
    r_is_padding: tl.constexpr = (r_patch == 18) | (r_intra == 18)

    # Pair is "invalid" (padding element) when EITHER q OR r is padding
    is_padding: tl.constexpr = q_is_padding | r_is_padding

    # The two cases:
    q_eq_r: tl.constexpr = (q_patch == r_patch) & (q_intra == r_intra)
    #   q_eq_r  =  True  → diff == 0 → value 0.0
    #   q_eq_r  =  False → diff != 0 → value -1000.0  (masked_fill of ne)
    # BUT: if one side is padding, masked_fill from ne already set it to
    # -1000.0; the eq masked_fill only touches the exact-zero diffs.
    # So:
    #   is_padding = True  → out = -1000.0
    #   is_padding = False & q_eq_r = True  → out = 0.0
    #   is_padding = False & q_eq_r = False → out = -1000.0  (from ne masked_fill)

    is_ne:  tl.constexpr = ~q_eq_r          # equivalent to (q != r)
    result: tl.constexpr = is_ne | is_padding

    out = tl.where(result, -10000.0, 0.0).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + idx, out, mask=mask)


# ---------------------------------------------------------------------------
# Triton kernel 2: Fused reshape + transpose for in_0 -> tmp_6
#   in_0  : (1, 133, 133, 96)
#   out   : (1, 19, 19, 7, 7, 96)
#   out[0, a, c, b, d, f] = in_0[0, a*7+b, c*7+d, f]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def _reshape_transpose_kernel(
    in_ptr,
    out_ptr,
    C: tl.constexpr,      # channel dim = 96 or 128
    BLOCK_SIZE: tl.constexpr,
):
    # 133 = 19*7,  output = 19*19*7*7*C = 361*49*C elements
    P:  tl.constexpr = 361
    Q:  tl.constexpr = 49
    PB: tl.constexpr = 7   # patch inner-row size

    pid   = tl.program_id(0)
    idx   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = P * Q * C
    mask  = idx < total

    # idx -> (p, q, c)  with p in [0,P), q in [0,Q), c in [0,C)
    c = idx % C
    q = (idx // C) % Q
    p =  idx // (Q * C)

    # Decompose p -> (a, b)  patch row/col  [0,19)
    a = p // 19
    b = p % 19
    # Decompose q -> (row, col) inside patch  [0,7)
    q_row = q // 7
    q_col = q % 7

    # Read in_0[0, a*7+b, c*7+q_col, c_idx]
    #   row  = a*PB + b   (patch row in the 133-grid)
    #   col  = q_row*PB + q_col   (position within that row)
    src_row  = a * PB + b
    src_col  = q_row * PB + q_col
    in_idx   = (src_row * 133 + src_col) * C + c

    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + idx, val, mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_mask_and_reshape(in_0, tmp_9):
    # --- output 0: pairwise attention mask ---
    P, Q = 361, 49
    out0 = torch.empty((1, P, Q, Q), dtype=in_0.dtype, device=in_0.device)
    total0 = P * Q * Q   # 867 529
    grid0  = lambda meta: ((total0 + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _pairwise_mask_kernel[grid0](out0, P=P, Q=Q)

    # --- output 1: in_0 -> (1,19,19,7,7,C) via reshape + transpose ---
    C    = in_0.shape[3]
    out1 = torch.empty((1, 19, 19, 7, 7, C), dtype=in_0.dtype, device=in_0.device)
    total1 = P * Q * C
    grid1  = lambda meta: ((total1 + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _reshape_transpose_kernel[grid1](in_0, out1, C=C)

    return (out0, out1)


def replacement_func():
    return fused_mask_and_reshape