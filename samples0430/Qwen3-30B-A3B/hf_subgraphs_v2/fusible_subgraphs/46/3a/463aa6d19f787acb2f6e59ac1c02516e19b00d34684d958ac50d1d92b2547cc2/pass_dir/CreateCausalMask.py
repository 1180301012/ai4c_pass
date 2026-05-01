import torch
import triton
import triton.language as tl

def pattern():
    n = 21
    tmp_1 = torch.arange(0, n, device=device(type='cuda', index=0))
    tmp_2 = torch.full((n, n), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(n, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6
    return tmp_7

def replacement_args(n):
    return (n,)

@triton.jit
def causal_mask_4d_kernel(
    mask_ptr,
    n: tl.int32,
    BLOCK_SIZE: tl.constexpr = 64
):
    start_idx = tl.program_id(0) * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, n * n)
    for idx in range(start_idx, end_idx):
        i = idx // n
        j = idx % n
        if j > i:
            value = -3.4028234663852886e+38
        else:
            value = 0.0
        tl.store(mask_ptr + i * n + j, value)

@torch.fx.wrap
def causal_mask_4d(n):
    mask = torch.empty((1, 1, n, n), dtype=torch.float32, device='cuda')
    num_elements = n * n
    BLOCK_SIZE = 64
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    causal_mask_4d_kernel[(num_blocks,)](mask, n, BLOCK_SIZE)
    return mask

def replacement_func():
    return causal_mask_4d