import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Shared kernel: general B*C programs approach
# Grid: (B * C,)  — each program handles one (b,c) scale + its H*W elements
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4,  num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def fused_linear_sigmoid_mul_kernel(
    in2_ptr,    # [B, K]    - contiguous
    weight_ptr, # [C, K]    - contiguous
    bias_ptr,   # [C]       - contiguous
    in3_ptr,    # [B,C,HW]  - contiguous (HW = H*W)
    out_ptr,    # [B,C,HW]  - same layout
    B, C, K: tl.constexpr, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid: (B * C,)
    Each program: compute scale = sigmoid(dot(in2[b,:], weight[c,:]) + bias[c])
                  then multiply scale into in3[b,c,:] block.
    """
    pid = tl.program_id(0)   # flattened (b,c) index
    b   = pid // C
    c   = pid  % C

    # ---- scalar computation -----------------------------------------------
    k_offs   = tl.arange(0, K)
    in2_vals = tl.load(in2_ptr    + b * K + k_offs).to(tl.float32)
    w_vals   = tl.load(weight_ptr + c * K + k_offs).to(tl.float32)
    bias_val = tl.load(bias_ptr   + c       ).to(tl.float32)

    linear_out = tl.sum(in2_vals * w_vals, axis=0) + bias_val
    scale      = tl.sigmoid(linear_out)

    # ---- vectorised multiply over H*W block --------------------------------
    bc       = pid
    hw_offs  = tl.arange(0, BLOCK_SIZE)
    mask     = hw_offs < HW
    base     = bc * HW

    vals   = tl.load(in3_ptr + base + hw_offs, mask=mask, other=0.0).to(tl.float32)
    result = (vals * scale).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + base + hw_offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Specialised B=1 kernel: 1-D grid with sequential channel loop.
# Loads in2 once; iterates over C channels each covering BLOCK_HW hw elements.
# Fixed config chosen so Grid=(C, HW//BLOCK_HW) gives enough programs for
# good GPU occupancy without excess launch overhead.
# ---------------------------------------------------------------------------
@triton.jit
def fused_linear_sigmoid_mul_B1_kernel(
    in2_ptr,    # [K]     in3_ptr: [C, HW]   out_ptr: [C, HW]
    weight_ptr, # [C, K]
    bias_ptr,   # [C]
    in3_ptr,
    out_ptr,
    C, K: tl.constexpr, HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Grid: (cdiv(HW, BLOCK_HW),)
    pid0 = spatial block b_hw  — loads in2 once for all channels
    Sequential loop over C channels: one program, many channels.
    """
    pid_hw  = tl.program_id(0)
    hw_base = pid_hw * BLOCK_HW
    hw_offs = hw_base + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
    hw_mask = hw_offs < HW

    # Load in2 as a contiguous vector (K=8, loaded once, kept in registers)
    k_offs  = tl.arange(0, K)
    in2_row = tl.load(in2_ptr + k_offs).to(tl.float32)   # [K]

    for c in range(C):
        w_vals   = tl.load(weight_ptr + c * K + k_offs).to(tl.float32)  # [K]
        bias_val = tl.load(bias_ptr   + c      ).to(tl.float32)
        linear   = tl.sum(in2_row * w_vals, axis=0) + bias_val
        scale    = tl.sigmoid(linear)                    # scalar float32

        base   = c * HW
        vals   = tl.load(in3_ptr + base + hw_offs,
                         mask=hw_mask, other=0.0).to(tl.float32)
        result = (vals * scale).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + base + hw_offs, result, mask=hw_mask)


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_linear_sigmoid_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C]
    in_1 : weight [C, K]
    in_2 : input  [B, K]
    in_3 : feats  [B, C, H, W]
    """
    B  = in_2.shape[0]
    C  = in_0.shape[0]
    K  = in_1.shape[1]
    HW = in_3.shape[2] * in_3.shape[3]

    out = torch.empty_like(in_3)

    if B == 1:
        # Special path: sequential channel loop, fixed BLOCK_HW=1024, num_warps=16
        # Grid=(4,) for HW=4096 → 4 programs, each looping over all 64 channels.
        _BLOCK_HW  = 1024
        _NUM_WARPS = 16
        fused_linear_sigmoid_mul_B1_kernel[
            (triton.cdiv(HW, _BLOCK_HW),)
        ](
            in_2, in_1, in_0, in_3, out,
            C, K, HW,
            BLOCK_HW=_BLOCK_HW,
            num_warps=_NUM_WARPS,
        )
    else:
        # General path: B*C programs, each handles one (b,c) slice
        # BLOCK_SIZE and num_warps selected by autotune (key=['HW'])
        fused_linear_sigmoid_mul_kernel[lambda meta: (B * C,)](
            in_2, in_1, in_0, in_3, out,
            B, C, K, HW,
        )

    return out