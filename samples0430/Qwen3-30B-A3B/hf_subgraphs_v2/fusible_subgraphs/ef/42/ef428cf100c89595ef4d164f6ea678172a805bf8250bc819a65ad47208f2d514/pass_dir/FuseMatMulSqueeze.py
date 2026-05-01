import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def matmul_squeeze_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    k,  # inner dimension (249)
    m,  # output dimension (64)
    BLOCK_SIZE: tl.constexpr
):
    j = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.bfloat16)
    for start in range(0, k, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, k)
        in_0_chunk = tl.cast(tl.load(in_0_ptr + start, mask=start < k, other=0.0), dtype=tl.bfloat16)
        in_1_chunk = tl.cast(tl.load(in_1_ptr + start * m + j, mask=start < k, other=0.0), dtype=tl.bfloat16)
        acc = acc + in_0_chunk * in_1_chunk
    tl.store(out_ptr + j, acc)

@torch.fx.wrap
def optimized_matmul_squeeze(x, y):
    batch, seq, d_model = x.shape
    batch_y, d_model, d_head = y.shape
    assert batch == batch_y == 1 and seq == 1
    out = torch.empty((batch, d_head), device=x.device, dtype=x.dtype)
    grid = (d_head,)
    BLOCK_SIZE = 64
    matmul_squeeze_kernel[grid](
        x,
        y,
        out,
        d_model,
        d_head,
        BLOCK_SIZE
    )
    return out

def replacement_func():
    return optimized_matmul_squeeze