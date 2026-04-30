import torch
import triton
import triton.language as tl

PATCH_H = 384
PATCH_W = 384
CHANNELS = 3
BLOCK_H = 8
BLOCK_W = 64
TILES_H = PATCH_H // BLOCK_H
TILES_W = PATCH_W // BLOCK_W
TILE_COUNT = TILES_H * TILES_W
PATCH_ELEMS = CHANNELS * PATCH_H * PATCH_W


@triton.jit
def _extract_patch_tiles_kernel(
    src_ptr,
    out_ptr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    STRIDE: tl.constexpr,
    PATCHES_PER_ROW: tl.constexpr,
    OUT_PATCH_BASE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_patch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_tile = tl.program_id(2)

    tile_h_idx = pid_tile // 6
    tile_w_idx = pid_tile % 6

    offs_h = tile_h_idx * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    offs_w = tile_w_idx * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    patch_row = pid_patch // PATCHES_PER_ROW
    patch_col = pid_patch % PATCHES_PER_ROW

    src_h = patch_row * STRIDE + offs_h
    src_w = patch_col * STRIDE + offs_w

    src_offsets = pid_c * (H_IN * W_IN) + src_h * W_IN + src_w
    values = tl.load(src_ptr + src_offsets)

    out_patch = OUT_PATCH_BASE + pid_patch
    out_offsets = out_patch * 442368 + pid_c * 147456 + offs_h * 384 + offs_w
    tl.store(out_ptr + out_offsets, tl.cast(values, tl.float16))


@triton.jit
def _copy_patch_tiles_kernel(
    src_ptr,
    out_ptr,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_stride_3,
    OUT_PATCH_BASE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_patch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_tile = tl.program_id(2)

    tile_h_idx = pid_tile // 6
    tile_w_idx = pid_tile % 6

    offs_h = tile_h_idx * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    offs_w = tile_w_idx * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    src_offsets = (
        pid_patch * src_stride_0
        + pid_c * src_stride_1
        + offs_h * src_stride_2
        + offs_w * src_stride_3
    )
    out_patch = OUT_PATCH_BASE + pid_patch
    out_offsets = out_patch * 442368 + pid_c * 147456 + offs_h * 384 + offs_w

    values = tl.load(src_ptr + src_offsets)
    tl.store(out_ptr + out_offsets, tl.cast(values, tl.float16))


@torch.fx.wrap
def depthpro_extract_large(in_2):
    out = torch.empty((25, CHANNELS, PATCH_H, PATCH_W), device=in_2.device, dtype=torch.float16)
    _extract_patch_tiles_kernel[(25, CHANNELS, TILE_COUNT)](
        in_2,
        out,
        H_IN=1536,
        W_IN=1536,
        STRIDE=288,
        PATCHES_PER_ROW=5,
        OUT_PATCH_BASE=0,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


@torch.fx.wrap
def depthpro_extract_small(in_1):
    out = torch.empty((9, CHANNELS, PATCH_H, PATCH_W), device=in_1.device, dtype=torch.float16)
    _extract_patch_tiles_kernel[(9, CHANNELS, TILE_COUNT)](
        in_1,
        out,
        H_IN=768,
        W_IN=768,
        STRIDE=192,
        PATCHES_PER_ROW=3,
        OUT_PATCH_BASE=0,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


@torch.fx.wrap
def depthpro_cat_to_fp16(tmp_5, tmp_2, in_0):
    total = tmp_5.shape[0] + tmp_2.shape[0] + in_0.shape[0]
    out = torch.empty((total, CHANNELS, PATCH_H, PATCH_W), device=in_0.device, dtype=torch.float16)

    s50, s51, s52, s53 = tmp_5.stride()
    s20, s21, s22, s23 = tmp_2.stride()
    i00, i01, i02, i03 = in_0.stride()

    _copy_patch_tiles_kernel[(tmp_5.shape[0], CHANNELS, TILE_COUNT)](
        tmp_5,
        out,
        s50,
        s51,
        s52,
        s53,
        OUT_PATCH_BASE=0,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    _copy_patch_tiles_kernel[(tmp_2.shape[0], CHANNELS, TILE_COUNT)](
        tmp_2,
        out,
        s20,
        s21,
        s22,
        s23,
        OUT_PATCH_BASE=tmp_5.shape[0],
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    _copy_patch_tiles_kernel[(in_0.shape[0], CHANNELS, TILE_COUNT)](
        in_0,
        out,
        i00,
        i01,
        i02,
        i03,
        OUT_PATCH_BASE=tmp_5.shape[0] + tmp_2.shape[0],
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


@torch.fx.wrap
def depthpro_whole_to_fp16(in_0, in_1, in_2):
    out = torch.empty((35, CHANNELS, PATCH_H, PATCH_W), device=in_0.device, dtype=torch.float16)

    _extract_patch_tiles_kernel[(25, CHANNELS, TILE_COUNT)](
        in_2,
        out,
        H_IN=1536,
        W_IN=1536,
        STRIDE=288,
        PATCHES_PER_ROW=5,
        OUT_PATCH_BASE=0,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    _extract_patch_tiles_kernel[(9, CHANNELS, TILE_COUNT)](
        in_1,
        out,
        H_IN=768,
        W_IN=768,
        STRIDE=192,
        PATCHES_PER_ROW=3,
        OUT_PATCH_BASE=25,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    _extract_patch_tiles_kernel[(1, CHANNELS, TILE_COUNT)](
        in_0,
        out,
        H_IN=384,
        W_IN=384,
        STRIDE=0,
        PATCHES_PER_ROW=1,
        OUT_PATCH_BASE=34,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


@torch.fx.wrap
def depthpro_dispatch(*args):
    route = args[-1]
    if route == "whole":
        return depthpro_whole_to_fp16(args[0], args[1], args[2])
    if route == "large":
        return depthpro_extract_large(args[0])
    if route == "small":
        return depthpro_extract_small(args[0])
    if route == "cat":
        return depthpro_cat_to_fp16(args[0], args[1], args[2])
    raise RuntimeError(f"Unknown route: {route}")


def replacement_func():
    return depthpro_dispatch