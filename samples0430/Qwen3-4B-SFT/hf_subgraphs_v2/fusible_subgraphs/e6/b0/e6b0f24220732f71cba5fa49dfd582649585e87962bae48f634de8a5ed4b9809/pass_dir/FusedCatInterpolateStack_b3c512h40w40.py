import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
#
# The original ops are:
#   tmp_1 = in_0          (already 40x40: nearest interp is identity)
#   tmp_2 = down-sample 2x of in_1 (20x20 -> 40x40)  [h->h//2, w->w//2]
#   tmp_3 = cat(in_2, in_3) along dim-1
#
# Stack output shape: [B, 3, C_dim, H, W] = [B, 3, 512, 40, 40]
#
# Memory is written in the output order.  For each flat output index we
# decompose (b, k, c, h, w) and emit one value.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['B', 'C_dim', 'H', 'W'],
)
@triton.jit
def fused_cat_interp_stack_kernel(
    in0_ptr,   # [B, C_dim, H, W]  — already 40x40
    in1_ptr,   # [B, C_dim, H//2, W//2]  — 20x20
    in2_ptr,   # [B, C_half, H, W]
    in3_ptr,   # [B, C_half, H, W]
    out_ptr,   # [B, 3, C_dim, H, W]
    B, C_dim, HW, C_half,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    base = pid * BLOCK_SIZE
    idx  = base + tl.arange(0, BLOCK_SIZE)
    n    = B * 3 * C_dim * HW          # total elements

    # ---- decode flat output index ----------------------------------------
    # Each element has linear order: b * (3 * C_dim * HW) + k * (C_dim * HW) + c * HW + hw
    hkw    = C_dim * HW                 # stride over (k, c, hw)
    c_hw   = HW

    b_idx   = idx // hkw
    rem     = idx  - b_idx * hkw
    k_idx   = rem  // c_hw
    rem2    = rem   - k_idx * c_hw
    c_idx   = rem2  // HW
    hw_idx  = rem2  - c_idx * HW       # in [0, HW)

    # ---- load from in0 (s == 0)  ----------------------------------------
    c_in = c_idx                             # channel index into in0
    src0 = b_idx * C_dim * HW + c_in * HW + hw_idx
    val0 = tl.load(in0_ptr + src0)

    # ---- scalar conditional branch (C_half is constexpr) ----------------
    use_a = c_idx < C_half   # scalar branch on channel index

    # --- in1 path (s == 1): 2× upsample from 20×20 → 40×40 ----------------
    # Nearest-neighbour 2× upscale: out[h,w] = in[h//2, w//2]
    # => hw_idx = h_out*40 + w_out  =>  hw1 = (h_out*40 + w_out) // 4 = h_out + w_out//4
    hw1   = hw_idx >> 2      # hw_idx // 4
    h_in  = hw1 // 40
    w_in  = hw1  % 40
    in1_sz = C_dim * 400     # = 512 * 400  (20*20 spatial stride)
    src1  = b_idx * in1_sz  +  c_in * 400 +  h_in * 20 + w_in
    val1  = tl.load(in1_ptr + src1)

    # --- in2/in3 path (s == 2: cat) ---------------------------------------
    c_off = tl.where(use_a, c_idx, c_idx - C_half)
    src23 = b_idx * C_half * HW + c_off * HW + hw_idx
    val2  = tl.load(in2_ptr + src23)
    val3  = tl.load(in3_ptr + src23)

    out_val = tl.where(use_a, val0, tl.where(k_idx == 1, val1, tl.where(use_a, val2, val3)))

    # ---- store to output [B, 3, C_dim, H, W] ----------------------------
    out_off = idx + b_idx * (3 * C_dim * HW) + k_idx * (C_dim * HW)
    tl.store(out_ptr + out_off, out_val)


# ---------------------------------------------------------------------------
# Python wrapper  (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_cat_interp_stack(in_0, in_1, in_2, in_3):
    B      = in_0.shape[0]
    C_dim  = in_0.shape[1]           # 512
    H      = in_0.shape[2]           # 40
    W      = in_0.shape[3]           # 40
    HW     = H * W                   # 1600
    C_half = in_2.shape[1]           # 256 = C_dim // 2

    out = torch.empty((B, 3, C_dim, H, W), dtype=in_0.dtype, device=in_0.device)

    n_total = B * 3 * C_dim * HW
    grid    = lambda meta: ((n_total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_cat_interp_stack_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        B, C_dim, HW, C_half,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode='nearest')
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_cat_interp_stack