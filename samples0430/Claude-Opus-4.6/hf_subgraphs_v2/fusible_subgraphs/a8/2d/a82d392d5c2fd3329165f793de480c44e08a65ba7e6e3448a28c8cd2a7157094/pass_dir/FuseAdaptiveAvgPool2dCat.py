import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_avgpool_cat_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    n_bc,  # B * 60
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: [B*60, 3] - each program handles 256 elements of one (batch, channel)
    bc_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)

    b = bc_idx // 60
    c = bc_idx % 60

    # Spatial offsets (768 = 3 * 256, no masking needed)
    spatial_off = tile_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Decompose spatial offset into h, w
    h = spatial_off // 24
    w = spatial_off - h * 24

    # Pool path: in_0[b, c, 2h, 2w] - masked loads skip memory access when c >= 20
    pool_mask = c < 20
    h_in = h * 2
    w_in = w * 2
    in0_addr = b * 61440 + c * 3072 + h_in * 48 + w_in

    val_00 = tl.load(in_0_ptr + in0_addr, mask=pool_mask, other=0.0)
    val_01 = tl.load(in_0_ptr + in0_addr + 1, mask=pool_mask, other=0.0)
    val_10 = tl.load(in_0_ptr + in0_addr + 48, mask=pool_mask, other=0.0)
    val_11 = tl.load(in_0_ptr + in0_addr + 49, mask=pool_mask, other=0.0)
    pool_val = (val_00 + val_01 + val_10 + val_11) * 0.25

    # Copy path: in_1[b, c-20, h, w] - masked loads skip when c < 20
    copy_mask = c >= 20
    c_copy = c - 20
    in1_addr = b * 30720 + c_copy * 768 + spatial_off
    copy_val = tl.load(in_1_ptr + in1_addr, mask=copy_mask, other=0.0)

    # Select result based on channel
    result = tl.where(pool_mask, pool_val, copy_val)

    # Store to output [B, 60, 32, 24]
    out_addr = b * 46080 + c * 768 + spatial_off
    tl.store(out_ptr + out_addr, result)


@torch.fx.wrap
def fused_avgpool_cat(in_0, in_1):
    B = in_0.shape[0]
    out = torch.empty(B, 60, 32, 24, dtype=in_0.dtype, device=in_0.device)
    n_bc = B * 60
    grid = (n_bc, 3)
    fused_avgpool_cat_kernel[grid](in_0, in_1, out, n_bc, BLOCK_SIZE=256)
    return out


def replacement_func():
    return fused_avgpool_cat