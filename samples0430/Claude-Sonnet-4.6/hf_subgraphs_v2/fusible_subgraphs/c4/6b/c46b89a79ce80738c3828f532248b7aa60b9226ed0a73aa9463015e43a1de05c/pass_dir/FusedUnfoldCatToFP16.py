"""
Fused pass: unfold + permute + reshape + cat + to(float16)
Handles both bfloat16 and float16 inputs.

Pipeline:
  in_1 [1,3,768,768]   --unfold(384,192)--> 9 patches  --> out[25:34]
  in_2 [1,3,1536,1536] --unfold(384,288)--> 25 patches --> out[0:25]
  in_0 [1,3,384,384]                      --> 1 patch   --> out[34:35]
  Output: [35,3,384,384] float16
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: extract 25 patches from in_2 [1,3,1536,1536] -> out[0:25]
#   stride=(288,288), kernel=(384,384), patch grid 5x5
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def copy_in2_patches_kernel(
    src_ptr,   # [1, 3, 1536, 1536]
    out_ptr,   # [35, 3, 384, 384] float16
    BLOCK_SIZE: tl.constexpr,
):
    """Copy 25 patches (5x5 grid, stride=288) from 1536x1536 image -> out[0:25]."""
    patch_id = tl.program_id(0)   # 0..24
    blk_id   = tl.program_id(1)

    offsets = blk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < 3 * 384 * 384

    # Decode (c, h, w) from flat offset within a patch
    c  = offsets // (384 * 384)
    hw = offsets % (384 * 384)
    h  = hw // 384
    w  = hw % 384

    # Patch location in the 5x5 grid (stride=288)
    ph = patch_id // 5
    pw = patch_id % 5
    src_h = ph * 288 + h
    src_w = pw * 288 + w

    # Flat offset into src [1, 3, 1536, 1536] (batch=0)
    src_off = c * (1536 * 1536) + src_h * 1536 + src_w
    # Output slot: patch_id * PATCH_ELEMENTS + offset
    out_off = patch_id * (3 * 384 * 384) + offsets

    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    val = val.to(tl.float16)
    tl.store(out_ptr + out_off, val, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: extract 9 patches from in_1 [1,3,768,768] -> out[25:34]
#   stride=(192,192), kernel=(384,384), patch grid 3x3
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def copy_in1_patches_kernel(
    src_ptr,   # [1, 3, 768, 768]
    out_ptr,   # [35, 3, 384, 384] float16
    BLOCK_SIZE: tl.constexpr,
):
    """Copy 9 patches (3x3 grid, stride=192) from 768x768 image -> out[25:34]."""
    patch_id = tl.program_id(0)   # 0..8
    blk_id   = tl.program_id(1)

    offsets = blk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < 3 * 384 * 384

    c  = offsets // (384 * 384)
    hw = offsets % (384 * 384)
    h  = hw // 384
    w  = hw % 384

    # Patch location in the 3x3 grid (stride=192)
    ph = patch_id // 3
    pw = patch_id % 3
    src_h = ph * 192 + h
    src_w = pw * 192 + w

    src_off = c * (768 * 768) + src_h * 768 + src_w
    # Output slots 25..33
    out_off = (25 + patch_id) * (3 * 384 * 384) + offsets

    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    val = val.to(tl.float16)
    tl.store(out_ptr + out_off, val, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3: copy in_0 [1,3,384,384] -> out[34]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def copy_in0_patch_kernel(
    src_ptr,   # [1, 3, 384, 384]
    out_ptr,   # [35, 3, 384, 384] float16
    BLOCK_SIZE: tl.constexpr,
):
    """Copy in_0 (contiguous 384x384 image) -> out[34]."""
    blk_id = tl.program_id(1)   # program_id(0) == 0 always (1 patch)

    offsets = blk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < 3 * 384 * 384

    src_off = offsets                               # in_0 is contiguous
    out_off = 34 * (3 * 384 * 384) + offsets        # slot 34

    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    val = val.to(tl.float16)
    tl.store(out_ptr + out_off, val, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper: allocate output and launch the three kernels
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_unfold_cat_to_fp16(in_0, in_1, in_2):
    """
    Fused replacement for:
      unfold(in_1,384,192) → permute → reshape → tmp_2 [9,3,384,384]
      unfold(in_2,384,288) → permute → reshape → tmp_5 [25,3,384,384]
      cat([tmp_5, tmp_2, in_0]) → to(float16)
    """
    out = torch.empty((35, 3, 384, 384), dtype=torch.float16, device=in_0.device)

    PATCH_ELEMENTS = 3 * 384 * 384  # 442368

    grid_in2 = lambda meta: (25, PATCH_ELEMENTS // meta['BLOCK_SIZE'])
    copy_in2_patches_kernel[grid_in2](in_2, out)

    grid_in1 = lambda meta: (9, PATCH_ELEMENTS // meta['BLOCK_SIZE'])
    copy_in1_patches_kernel[grid_in1](in_1, out)

    grid_in0 = lambda meta: (1, PATCH_ELEMENTS // meta['BLOCK_SIZE'])
    copy_in0_patch_kernel[grid_in0](in_0, out)

    return out


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_unfold_cat_to_fp16