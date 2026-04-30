import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROWS': 1}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 2}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 4}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 8}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 1}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 2}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 4}, num_warps=8),
    ],
    key=['num_rows', 'cols'],
)
@triton.jit
def _silu_split_kernel(
    input_ptr,
    out_0_ptr,
    out_1_ptr,
    out_2_ptr,
    num_rows,
    cols,
    BLOCK_ROWS: tl.constexpr,
):
    """
    Fused SiLU + split kernel.
    Each program handles BLOCK_ROWS rows and all 3 chunks.
    in_1 layout: [B, 17, 1152]  (row = b*17 + r, col = j)
    out_0 layout: [B, 17, 512]  (chunk 0)
    out_1 layout: [B, 17, 512]  (chunk 1)
    out_2 layout: [B, 17, 1, 128] (chunk 2, unsqueeze already applied via shape)
    """
    pid_row = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    # Row range handled by this program
    row_start = pid_row * BLOCK_ROWS
    rows = row_start + tl.arange(0, BLOCK_ROWS)
    row_valid = rows < num_rows

    # Chunk column ranges
    if pid_chunk == 0:
        # chunk 0: columns [0, 512)
        c_start = 0
        c_offsets = c_start + tl.arange(0, 512)
    elif pid_chunk == 1:
        # chunk 1: columns [512, 1024)
        c_start = 512
        c_offsets = c_start + tl.arange(0, 512)
    else:
        # chunk 2: columns [1024, 1152)
        c_start = 1024
        c_offsets = c_start + tl.arange(0, 128)

    # Load input elements: input[rows, j]
    input_offsets = rows[:, None] * cols + c_offsets[None, :]
    load_mask = row_valid[:, None] & (c_offsets[None, :] < cols)
    x = tl.load(input_ptr + input_offsets, mask=load_mask, other=0.0)

    # SiLU: x * sigmoid(x)
    silu_x = x * tl.sigmoid(x)

    # Route to the correct output tensor and store
    if pid_chunk == 0:
        # out_0 shape [B, 17, 512], last-dim stride = 1
        out_offsets = rows[:, None] * 512 + c_offsets[None, :]
        tl.store(out_0_ptr + out_offsets, silu_x, mask=load_mask)
    elif pid_chunk == 1:
        # out_1 shape [B, 17, 512], last-dim stride = 1
        out_offsets = rows[:, None] * 512 + c_offsets[None, :]
        tl.store(out_1_ptr + out_offsets, silu_x, mask=load_mask)
    else:
        # out_2 shape [B, 17, 1, 128]; the extra dim is size-1 so the stride
        # of the second-to-last dim is 128 — same as if it were [B,17,128].
        # Writing at row*128+c is therefore identical to writing out_2[b,r,0,c].
        out_offsets = rows[:, None] * 128 + c_offsets[None, :]
        tl.store(out_2_ptr + out_offsets, silu_x, mask=load_mask)


@torch.fx.wrap
def fused_silu_split(in_0, in_1):
    """
    Fused replacement for:
        tmp_1 = silu(in_1, inplace=True)
        split = torch.split(tmp_1, [512,512,128], dim=2)
        tmp_7 = in_0[(None,None,:)]
        return (tmp_7, split[0], split[2].unsqueeze(2), split[1])
    """
    B = in_1.shape[0]
    H = in_1.shape[1]
    num_rows = B * H
    cols = in_1.shape[2]          # 1152

    # Allocate outputs
    out_0 = torch.empty((B, H, 512),       dtype=in_1.dtype, device=in_1.device)
    out_1 = torch.empty((B, H, 512),       dtype=in_1.dtype, device=in_1.device)
    # chunk 2 unsqueezed along dim 2: [B, H, 1, 128]
    out_2 = torch.empty((B, H, 1, 128),    dtype=in_1.dtype, device=in_1.device)
    # tmp_7 = in_0[(None, None, :)]: [2, 1, 1, 128]
    out_3 = in_0[(None, None, slice(None, None, None))]

    grid = lambda meta: (triton.cdiv(num_rows, meta['BLOCK_ROWS']), 3)

    _silu_split_kernel[grid](
        in_1, out_0, out_1, out_2,
        num_rows, cols,
    )

    return (out_3, out_0, out_2, out_1)


def replacement_func():
    return fused_silu_split