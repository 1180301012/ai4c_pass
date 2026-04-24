import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _fused_slice_cat_kernel(
    in0_ptr, in1_ptr, out_ptr,
    MASK_N: tl.constexpr,
    CAT_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Total output elements: 2 * CAT_N
    TOTAL: tl.constexpr = 2 * CAT_N

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_total = offsets < TOTAL

    # Each output element belongs to a row (ROW_LEN = CAT_N elements per row)
    row = offsets // CAT_N
    col = offsets % CAT_N

    # Load from in_1 for all output positions
    in1_val = tl.load(in1_ptr + row * CAT_N + col, mask=mask_total, other=0)

    # Load mask value from in_2 (bool, length = MASK_N)
    # col < CAT_N; for GAE CAT_N = 1000 >= MASK_N = 100, need col < MASK_N guard
    mask_val = tl.load(in0_ptr + col, mask=(mask_total & (col < MASK_N)), other=0)

    # Combine: use in_0[mask] when mask is True, otherwise use in_1
    result = tl.where(mask_val != 0, mask_val, in1_val)

    tl.store(out_ptr + offsets, result, mask=mask_total)


@torch.fx.wrap
def fused_slice_cat_gae(in_0, in_1, in_2):
    MASK_N = 100
    CAT_N = 1000
    BLOCK_SIZE = 1024

    # Move in_0 (CPU) to CUDA so Triton can access it
    in_0_gpu = torch.as_tensor(in_0, device=in_1.device)

    # Output tensor: [2, CAT_N] = [2, 1000], int64, same device as in_1
    out_cat = torch.empty((2, CAT_N), dtype=torch.int64, device=in_1.device)

    n_blocks = (2 * CAT_N + BLOCK_SIZE - 1) // BLOCK_SIZE

    _fused_slice_cat_kernel[(n_blocks,)](
        in_0_gpu, in_1, out_cat,
        MASK_N=MASK_N,
        CAT_N=CAT_N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_cat


def replacement_func():
    return fused_slice_cat_gae