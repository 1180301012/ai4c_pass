import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def l2_normalize_kernel(
    input_ptr,
    output_ptr,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * num_cols

    # Phase 1: Compute L2 norm (sum of squares)
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

@torch.fx.wrap
def l2_normalize_fused(in_1):
    num_rows = in_1.shape[0]
    num_cols = in_1.shape[1]
    BLOCK_SIZE = 1024
    grid = (num_rows,)
    out = torch.empty_like(in_1)
    l2_normalize_kernel[grid](
        input_ptr=in_1,
        output_ptr=out,
        num_cols=num_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return l2_normalize_fused