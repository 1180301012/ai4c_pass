import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BC': 1, 'SPATIAL_TILE': 64, 'NUM_SPATIAL_TILES': 3}, num_warps=2),
        triton.Config({'BLOCK_BC': 2, 'SPATIAL_TILE': 64, 'NUM_SPATIAL_TILES': 3}, num_warps=2),
        triton.Config({'BLOCK_BC': 4, 'SPATIAL_TILE': 64, 'NUM_SPATIAL_TILES': 3}, num_warps=4),
        triton.Config({'BLOCK_BC': 8, 'SPATIAL_TILE': 64, 'NUM_SPATIAL_TILES': 3}, num_warps=4),
        triton.Config({'BLOCK_BC': 16, 'SPATIAL_TILE': 64, 'NUM_SPATIAL_TILES': 3}, num_warps=8),
        triton.Config({'BLOCK_BC': 32, 'SPATIAL_TILE': 64, 'NUM_SPATIAL_TILES': 3}, num_warps=8),
        triton.Config({'BLOCK_BC': 1, 'SPATIAL_TILE': 128, 'NUM_SPATIAL_TILES': 2}, num_warps=2),
        triton.Config({'BLOCK_BC': 2, 'SPATIAL_TILE': 128, 'NUM_SPATIAL_TILES': 2}, num_warps=2),
        triton.Config({'BLOCK_BC': 4, 'SPATIAL_TILE': 128, 'NUM_SPATIAL_TILES': 2}, num_warps=4),
        triton.Config({'BLOCK_BC': 8, 'SPATIAL_TILE': 128, 'NUM_SPATIAL_TILES': 2}, num_warps=4),
        triton.Config({'BLOCK_BC': 16, 'SPATIAL_TILE': 128, 'NUM_SPATIAL_TILES': 2}, num_warps=8),
        triton.Config({'BLOCK_BC': 32, 'SPATIAL_TILE': 128, 'NUM_SPATIAL_TILES': 2}, num_warps=8),
    ],
    key=['BC_total', 'HW'],
)
@triton.jit
def fused_relu_add_avg_pool_kernel(
    in0_ptr, in1_ptr, out_ptr,
    BC_total, HW,
    BLOCK_BC: tl.constexpr,
    SPATIAL_TILE: tl.constexpr,
    NUM_SPATIAL_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    bc_start = pid * BLOCK_BC
    bc_offsets = bc_start + tl.arange(0, BLOCK_BC)
    bc_mask = bc_offsets < BC_total

    sum_val = tl.zeros([BLOCK_BC], dtype=tl.float32)

    for tile_idx in range(NUM_SPATIAL_TILES):
        spatial_start = tile_idx * SPATIAL_TILE
        spatial_offsets = spatial_start + tl.arange(0, SPATIAL_TILE)
        spatial_mask = spatial_offsets < HW

        input_offsets = bc_offsets[:, None] * HW + spatial_offsets[None, :]
        load_mask = bc_mask[:, None] & spatial_mask[None, :]

        in0_vals = tl.load(in0_ptr + input_offsets, mask=load_mask, other=0.0).to(tl.float32)
        in1_vals = tl.load(in1_ptr + input_offsets, mask=load_mask, other=0.0).to(tl.float32)

        relu_in1 = tl.where(in1_vals > 0.0, in1_vals, 0.0)
        combined = in0_vals + relu_in1

        sum_val += tl.sum(combined, axis=1)

    avg = sum_val / HW

    tl.store(out_ptr + bc_offsets, avg, mask=bc_mask)


@torch.fx.wrap
def fused_relu_add_avg_pool(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    BC_total = B * C

    out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (triton.cdiv(BC_total, meta['BLOCK_BC']),)

    fused_relu_add_avg_pool_kernel[grid](
        in0_ptr=in_0, in1_ptr=in_1, out_ptr=out,
        BC_total=BC_total, HW=HW,
    )

    return out


def replacement_func():
    return fused_relu_add_avg_pool