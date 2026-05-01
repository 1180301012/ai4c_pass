import torch
import triton
import triton.language as tl

def pattern(in_2, device):
    # Match the exact pattern: create range, add 0, view, and comparison
    seq_len = in_2.shape[0]
    tmp_6 = torch.arange(seq_len, device=device)
    tmp_7 = tmp_6 + 0
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    return tmp_9

def replacement_args(in_2, device):
    return (in_2, device)

@triton.jit
def generate_mask_kernel(
    out_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block handles a BLOCK_SIZE x BLOCK_SIZE tile
    block_x = tl.program_id(0)
    block_y = tl.program_id(1)
    x = block_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = block_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_grid, y_grid = tl.meshgrid(x, y)
    mask = x_grid <= y_grid
    out_idx = x_grid * seq_len + y_grid
    tl.store(out_ptr + out_idx, mask)

@torch.fx.wrap
def generate_mask(in_2, device):
    seq_len = in_2.shape[0]
    out = torch.empty((seq_len, seq_len), dtype=torch.bool, device=device)
    BLOCK_SIZE = 32
    num_blocks_x = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_y = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    generate_mask_kernel[(num_blocks_x, num_blocks_y)](
        out_ptr=out,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return generate_mask