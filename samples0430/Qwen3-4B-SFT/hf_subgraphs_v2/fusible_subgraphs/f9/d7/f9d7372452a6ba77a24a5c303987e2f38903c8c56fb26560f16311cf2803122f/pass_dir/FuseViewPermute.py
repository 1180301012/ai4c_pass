import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 64}),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 32}),
    ],
    key=['total_elements'],
)
@triton.jit
def view_permute_kernel(
    in_ptr,
    out_ptr,
    total_elements,
    inner_stride,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Input shape: [BH, W] (flattened batch/spatial dims → BH rows)
    # Output shape: [W, BH]
    pid = tl.program_id(0)
    # Each program handles a BLOCK_H x BLOCK_W tile of the output
    # offset = w * BH + h  (linear index in output [W, BH])
    w_offset = pid * BLOCK_H          # output column index (dim W)
    h_offset = pid * BLOCK_W          # output row index (dim BH)
    w_offsets = w_offset + tl.arange(0, BLOCK_W)
    h_offsets = h_offset + tl.arange(0, BLOCK_H)

    # Output indices: row = h, col = w → offset = h * W + w
    # But stored as [BH, W]: offset = w * inner_stride + h
    output_offsets = w_offsets[:, None] * inner_stride + h_offsets[None, :]

    mask = (w_offsets[:, None] < total_elements % (inner_stride)) & (h_offsets[None, :] < total_elements // inner_stride)
    # We only have 'inner_stride' h-rows (BH rows)
    mask = ((w_offsets[:, None] < (total_elements // inner_stride)) &
            (h_offsets[None, :] < (total_elements % inner_stride)))

    vals = tl.load(in_ptr + output_offsets, mask=mask)
    out_offsets = (w_offsets[:, None] * inner_stride + h_offsets[None, :])
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@torch.fx.wrap
def fuse_view_permute(in_1):
    # in_1 shape: [1, 32, 64, 48]
    # view(1, 32, -1) → [1, 32, 3072]
    # permute(0, 2, 1) → [1, 3072, 32]
    # output shape: [1, W, BH] = [1, 3072, 32]
    B, C, H, W = in_1.shape
    total_elements = B * C * H * W
    output_shape = (B, H * W, C)
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    grid = lambda meta: (
        triton.cdiv(H * W, meta['BLOCK_W']) * triton.cdiv(C, meta['BLOCK_H']),
    )
    view_permute_kernel[grid](in_1, out, total_elements, H * W)
    return out


def pattern(in_1):
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return fuse_view_permute