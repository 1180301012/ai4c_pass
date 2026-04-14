import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Kernel 1: fused multi-input add + spatial mean (for 1/2/3 tensors)
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},    num_warps=2),
        triton.Config({'BLOCK_HW': 128},   num_warps=4),
        triton.Config({'BLOCK_HW': 256},   num_warps=4),
        triton.Config({'BLOCK_HW': 512},   num_warps=8),
        triton.Config({'BLOCK_HW': 1024},  num_warps=8),
        triton.Config({'BLOCK_HW': 2048},  num_warps=16),
        triton.Config({'BLOCK_HW': 4096},  num_warps=16),
        triton.Config({'BLOCK_HW': 8192},  num_warps=16),
        triton.Config({'BLOCK_HW': 16384}, num_warps=16),
    ],
    key=['HW', 'N_INPUTS', 'IS_FP16', 'IS_BF16'],
)
@triton.jit
def fused_add_mean_kernel(
    in0_ptr, in1_ptr, in2_ptr,
    out_ptr, mean_ptr,
    HW,
    N_INPUTS: tl.constexpr,
    IS_FP16:  tl.constexpr,
    IS_BF16:  tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program handles one (batch, channel) slice of HW spatial elements.
    Fuses: (optional multi-input addition) + (store result) + (spatial mean reduction).
    """
    bc   = tl.program_id(0)
    base = bc * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for i in range(0, tl.cdiv(HW, BLOCK_HW)):
        hw_off = i * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask   = hw_off < HW

        x = tl.load(in0_ptr + base + hw_off, mask=mask, other=0.0).to(tl.float32)

        if N_INPUTS >= 2:
            y = tl.load(in1_ptr + base + hw_off, mask=mask, other=0.0).to(tl.float32)
            x = x + y

        if N_INPUTS >= 3:
            z = tl.load(in2_ptr + base + hw_off, mask=mask, other=0.0).to(tl.float32)
            x = x + z

        # Store back in input dtype
        if IS_FP16:
            tl.store(out_ptr + base + hw_off, x.to(tl.float16), mask=mask)
        elif IS_BF16:
            tl.store(out_ptr + base + hw_off, x.to(tl.bfloat16), mask=mask)
        else:
            tl.store(out_ptr + base + hw_off, x, mask=mask)

        # Out-of-bounds lanes have other=0.0 so they don't affect the sum
        acc = acc + x

    mean_val = tl.sum(acc) / HW

    if IS_FP16:
        tl.store(mean_ptr + bc, mean_val.to(tl.float16))
    elif IS_BF16:
        tl.store(mean_ptr + bc, mean_val.to(tl.bfloat16))
    else:
        tl.store(mean_ptr + bc, mean_val)


def run_fused_add_mean(in_0, in_1, in_2, n_inputs):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    out      = torch.empty_like(in_0)
    mean_out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    is_fp16 = (in_0.dtype == torch.float16)
    is_bf16 = (in_0.dtype == torch.bfloat16)
    dummy   = in_0  # harmless placeholder for unused pointer args

    fused_add_mean_kernel[(BC,)](
        in_0,
        in_1 if in_1 is not None else dummy,
        in_2 if in_2 is not None else dummy,
        out,
        mean_out.view(-1),
        HW,
        N_INPUTS=n_inputs,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    return (out, mean_out)


# ──────────────────────────────────────────────────────────────────────────────
# Kernel 2: spatial mean-only (2-D tiling: BLOCK_BC channels × BLOCK_HW spatial)
# Processing BLOCK_BC channels per CTA drastically cuts launch overhead for
# cases with large C but small H*W (e.g. [1, 2048, 8, 8]).
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # Multi-channel-per-CTA configs (good for small/medium HW)
        triton.Config({'BLOCK_BC': 64, 'BLOCK_HW': 64},   num_warps=8),
        triton.Config({'BLOCK_BC': 32, 'BLOCK_HW': 64},   num_warps=4),
        triton.Config({'BLOCK_BC': 32, 'BLOCK_HW': 128},  num_warps=8),
        triton.Config({'BLOCK_BC': 32, 'BLOCK_HW': 256},  num_warps=8),
        triton.Config({'BLOCK_BC': 16, 'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_BC': 16, 'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_BC': 16, 'BLOCK_HW': 256},  num_warps=8),
        triton.Config({'BLOCK_BC': 8,  'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_BC': 8,  'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_BC': 8,  'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_BC': 4,  'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_BC': 4,  'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_BC': 4,  'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_BC': 2,  'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_BC': 2,  'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_BC': 2,  'BLOCK_HW': 2048}, num_warps=16),
        # Single-channel-per-CTA configs (good for large HW)
        triton.Config({'BLOCK_BC': 1,  'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_BC': 1,  'BLOCK_HW': 2048}, num_warps=16),
        triton.Config({'BLOCK_BC': 1,  'BLOCK_HW': 4096}, num_warps=16),
        triton.Config({'BLOCK_BC': 1,  'BLOCK_HW': 8192}, num_warps=16),
        triton.Config({'BLOCK_BC': 1,  'BLOCK_HW': 16384},num_warps=16),
    ],
    key=['HW', 'IS_FP16', 'IS_BF16'],
)
@triton.jit
def mean_only_kernel(
    in_ptr, mean_ptr,
    HW, BC,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_BC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    2-D tiled spatial mean.
    Each CTA handles BLOCK_BC consecutive (b,c) channels,
    iterating over BLOCK_HW spatial elements at a time.
    """
    pid     = tl.program_id(0)
    bc_offs = pid * BLOCK_BC + tl.arange(0, BLOCK_BC)   # [BLOCK_BC]
    bc_mask = bc_offs < BC

    # Accumulator: [BLOCK_BC, BLOCK_HW]
    acc = tl.zeros([BLOCK_BC, BLOCK_HW], dtype=tl.float32)

    for hw_start in range(0, tl.cdiv(HW, BLOCK_HW)):
        hw_offs  = hw_start * BLOCK_HW + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
        hw_mask  = hw_offs < HW
        mask_2d  = bc_mask[:, None] & hw_mask[None, :]             # [BLOCK_BC, BLOCK_HW]
        ptrs     = in_ptr + bc_offs[:, None] * HW + hw_offs[None, :]
        x        = tl.load(ptrs, mask=mask_2d, other=0.0).to(tl.float32)
        acc     += x

    # Sum over HW axis → [BLOCK_BC]
    mean_vals = tl.sum(acc, axis=1) / HW

    if IS_FP16:
        tl.store(mean_ptr + bc_offs, mean_vals.to(tl.float16), mask=bc_mask)
    elif IS_BF16:
        tl.store(mean_ptr + bc_offs, mean_vals.to(tl.bfloat16), mask=bc_mask)
    else:
        tl.store(mean_ptr + bc_offs, mean_vals, mask=bc_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Shared fast_mean wrapper
# ALL pass files import and return THIS exact object so they share the same
# replacement_func reference and the replacement_func_limit never drops them.
#
# Design:
#   pattern(x)    → returns (x, y)          [1 input, 2 observable outputs]
#   fast_mean(x)  → returns (x, mean_out)   [same 1-input / 2-output shape]
#   replacement_args(x) → (x,)              [1 arg, matching pattern inputs]
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fast_mean(x):
    """
    Computes spatial mean of x over dims (2,3) with keepdim=True via a
    2-D tiled Triton kernel.  Returns mean_out only.
    """
    B, C, H, W = x.shape
    HW = H * W
    BC = B * C

    mean_out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    is_fp16 = (x.dtype == torch.float16)
    is_bf16 = (x.dtype == torch.bfloat16)

    # Grid uses a lambda so autotune can see BLOCK_BC when computing CTA count
    mean_only_kernel[lambda meta: (triton.cdiv(BC, meta['BLOCK_BC']),)](
        x,
        mean_out.view(-1),
        HW, BC,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    return mean_out