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
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offset = tl.program_id(1) * BLOCK_SIZE
    col_ids = tl.arange(0, BLOCK_SIZE)
    mask = col_ids + col_offset < D

    # Load input segment
    x = tl.load(x_ptr + row_id * D + col_offset + col_ids, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    x_sq = x_fp32 * x_fp32
    sum_sq = tl.sum(x_sq, axis=0)
    norm = tl.sqrt(sum_sq)
    norm_bf16 = norm.to(tl.bfloat16)
    out = x / norm_bf16

    # Store output
    tl.store(out_ptr + row_id * D + col_offset + col_ids, out, mask=mask)

@torch.fx.wrap
def l2_normalize(x):
    B, D = x.shape
    BLOCK_SIZE = 1024
    num_blocks = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)

    grid = (B, num_blocks)
    l2_normalize_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        B=B,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return l2_normalize