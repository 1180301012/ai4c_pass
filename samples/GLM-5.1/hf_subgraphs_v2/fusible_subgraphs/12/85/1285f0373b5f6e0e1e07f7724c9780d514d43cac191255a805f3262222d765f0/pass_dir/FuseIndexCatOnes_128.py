import torch
import triton
import triton.language as tl
from torch import device as torch_device


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = tmp_0[slice(None, None, None), in_2]
    tmp_2 = torch.ops.aten.sym_size.int(tmp_1, 1)
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    tmp_10 = torch.sym_sum([128, tmp_2])
    tmp_11 = torch.ones((tmp_10,), dtype=torch.float32, device=torch_device(type='cuda'))
    return (tmp_9, tmp_11)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_rect_l")


@triton.jit
def process_mask_kernel(mask_ptr, count_ptr, indices_ptr, mask_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < mask_size
    # Load boolean mask values (True=1, False=0)
    values = tl.load(mask_ptr + offsets, mask=mask_valid, other=0).to(tl.int32)

    # Count True values
    count = tl.sum(values)
    tl.store(count_ptr, count)

    # Compute prefix sum to determine output positions for True values
    prefix = tl.cumsum(values, axis=0)  # 1-indexed cumulative sum
    # For True values, store the original index at position prefix-1
    store_pos = prefix - 1
    store_mask = (values == 1) & mask_valid
    tl.store(indices_ptr + store_pos, offsets.to(tl.int64), mask=store_mask)


@triton.jit
def fused_gather_cat_ones_kernel(
    in_0_ptr, in_1_ptr, indices_ptr,
    out_cat_ptr, out_ones_ptr,
    rows, N_in0_cols, N_in1_cols, num_true,
    total_cat_cols, total_ones_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # --- Cat output ---
    cat_total = rows * total_cat_cols
    cat_mask = offsets < cat_total

    # Decode 2D position from flat offset
    r = offsets // total_cat_cols
    c = offsets % total_cat_cols

    # Part 1: Gather from in_0 using true indices (columns 0..num_true-1)
    is_gather = (c < num_true) & cat_mask
    idx = tl.load(indices_ptr + c, mask=is_gather, other=0)
    in0_val = tl.load(in_0_ptr + r * N_in0_cols + idx, mask=is_gather, other=0)

    # Part 2: Copy from in_1 (columns num_true..total_cat_cols-1)
    is_copy = (c >= num_true) & cat_mask
    c_in1 = c - num_true
    in1_mask = is_copy & (c_in1 < N_in1_cols)
    in1_val = tl.load(in_1_ptr + r * N_in1_cols + c_in1, mask=in1_mask, other=0)

    # Combine and store
    cat_val = tl.where(c < num_true, in0_val, in1_val)
    tl.store(out_cat_ptr + offsets, cat_val, mask=cat_mask)

    # --- Ones output ---
    ones_mask = offsets < total_ones_size
    tl.store(out_ones_ptr + offsets, 1.0, mask=ones_mask)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, route):
    # Transfer in_0 to GPU (it's originally on CPU)
    in_0_gpu = in_0.to(in_1.device)

    mask_size = in_2.shape[0]
    rows = in_0_gpu.shape[0]
    N_in0_cols = in_0_gpu.shape[1]
    N_in1_cols = in_1.shape[1]

    # Step 1: Process mask - count True values and compute indices
    BLOCK_SIZE_MASK = 256  # >= max mask_size (128 or 100)
    count_tensor = torch.empty(1, dtype=torch.int32, device=in_1.device)
    indices_tensor = torch.empty(mask_size, dtype=torch.int64, device=in_1.device)

    process_mask_kernel[(1,)](
        mask_ptr=in_2, count_ptr=count_tensor, indices_ptr=indices_tensor,
        mask_size=mask_size, BLOCK_SIZE=BLOCK_SIZE_MASK,
    )

    num_true = count_tensor.item()

    # Step 2: Determine output sizes based on route
    total_cat_cols = num_true + N_in1_cols
    if route == "route_rect_l":
        total_ones_size = 128 + num_true
    elif route == "route_gae":
        total_ones_size = 1000 + num_true
    else:
        raise ValueError(f"Unknown route: {route}")

    # Step 3: Allocate outputs
    out_cat = torch.empty((rows, total_cat_cols), dtype=in_0_gpu.dtype, device=in_1.device)
    out_ones = torch.empty((total_ones_size,), dtype=torch.float32, device=in_1.device)

    # Step 4: Launch fused kernel
    cat_total = rows * total_cat_cols
    max_total = max(cat_total, total_ones_size)
    BLOCK_SIZE_MAIN = 1024
    num_programs = (max_total + BLOCK_SIZE_MAIN - 1) // BLOCK_SIZE_MAIN

    fused_gather_cat_ones_kernel[(num_programs,)](
        in_0_ptr=in_0_gpu, in_1_ptr=in_1, indices_ptr=indices_tensor,
        out_cat_ptr=out_cat, out_ones_ptr=out_ones,
        rows=rows, N_in0_cols=N_in0_cols, N_in1_cols=N_in1_cols, num_true=num_true,
        total_cat_cols=total_cat_cols, total_ones_size=total_ones_size,
        BLOCK_SIZE=BLOCK_SIZE_MAIN,
    )

    return (out_cat, out_ones)


def replacement_func():
    return kernel_wrapper