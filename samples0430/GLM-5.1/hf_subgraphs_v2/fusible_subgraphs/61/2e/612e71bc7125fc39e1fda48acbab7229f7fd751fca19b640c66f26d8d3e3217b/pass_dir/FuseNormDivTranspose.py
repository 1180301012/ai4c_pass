import torch
import triton
import triton.language as tl
from torch import device

def pattern(in_0, in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return (tmp_1, tmp_3)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def l2_normalize_kernel(
    input_ptr,
    output_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    row_offset = row_idx * num_cols

    # Phase 1: Compute L2 norm (sum of squares then sqrt)
    sum_sq = 0.0
    for block_start in range(0, num_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_cols
        data = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq = sum_sq + tl.sum(data * data)

    norm_val = tl.sqrt(sum_sq)

    # Phase 2: Divide each element by norm and store
    for block_start in range(0, num_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_cols
        data = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        result = data / norm_val
        tl.store(output_ptr + row_offset + offsets, result, mask=mask)

@triton.jit
def transpose_copy_kernel(
    input_ptr,
    output_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, data, mask=mask)

@torch.fx.wrap
def fused_norm_transpose(in_0, in_1):
    # L2 normalize in_1
    num_rows = in_1.shape[0]
    num_cols = in_1.shape[1]
    BLOCK_SIZE = 1024
    grid_norm = (num_rows,)
    out_norm = torch.empty_like(in_1)
    l2_normalize_kernel[grid_norm](
        input_ptr=in_1,
        output_ptr=out_norm,
        num_rows=num_rows,
        num_cols=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Transpose in_0: [1, N] -> [N, 1]
    num_rows_out = in_0.shape[1]
    num_cols_out = in_0.shape[0]
    total_elements = in_0.numel()
    BLOCK_SIZE_T = 1024
    num_programs = (total_elements + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    out_transpose = torch.empty((num_rows_out, num_cols_out), dtype=in_0.dtype, device=in_0.device)
    transpose_copy_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=out_transpose,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE_T,
    )

    return (out_norm, out_transpose)

def replacement_func():
    return fused_norm_transpose