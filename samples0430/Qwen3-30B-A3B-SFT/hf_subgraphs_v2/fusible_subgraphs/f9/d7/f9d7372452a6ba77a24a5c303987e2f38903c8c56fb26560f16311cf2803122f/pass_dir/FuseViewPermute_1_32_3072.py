import torch
import triton
import triton.language as tl


@triton.jit
def _view_permute_kernel(
    in_ptr, out_ptr,
    B, C, L,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Transpose [B, C, L] -> [B, L, C]
    # Grid: (B * cdiv(C, BLOCK_C), cdiv(L, BLOCK_L))
    bc = tl.program_id(0)   # index into C blocks
    bl = tl.program_id(1)   # index into L blocks
    b  = tl.program_id(2)   # batch index

    c_off = bc * BLOCK_C + tl.arange(0, BLOCK_C)
    l_off = bl * BLOCK_L + tl.arange(0, BLOCK_L)

    c_mask = c_off < C
    l_mask = l_off < L

    # Input:  [B, C, L] -> position = b*C*L + c*L + l
    # Output: [B, L, C] -> position = b*L*C + l*C + c
    in_base  = b * C * L
    out_base = b * L * C

    in_offsets  = in_base  + c_off[:, None] * L + l_off[None, :]  # [BC, BL]
    out_offsets = out_base + l_off[:, None] * C + c_off[None, :]  # [BL, BC]

    mask2d = c_mask[:, None] & l_mask[None, :]

    data = tl.load(in_ptr + in_offsets, mask=mask2d, other=0.0)
    tl.store(out_ptr + out_offsets, tl.trans(data), mask=tl.trans(mask2d))

@torch.fx.wrap
def view_permute_func(x):
    # x: [1, 32, 64, 48] -> [1, 3072, 32]
    B = x.shape[0]
    C = x.shape[1]
    L = x.numel() // (B * C)
    out = torch.empty((B, L, C), dtype=x.dtype, device=x.device)
    BLOCK_C = 32
    BLOCK_L = 128
    import math
    grid_c = math.ceil(C / BLOCK_C)
    grid_l = math.ceil(L / BLOCK_L)
    _view_permute_kernel[(grid_c, grid_l, B)](
        x, out, B, C, L,
        BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L,
    )
    return out


def pattern(in_1):
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return view_permute_func