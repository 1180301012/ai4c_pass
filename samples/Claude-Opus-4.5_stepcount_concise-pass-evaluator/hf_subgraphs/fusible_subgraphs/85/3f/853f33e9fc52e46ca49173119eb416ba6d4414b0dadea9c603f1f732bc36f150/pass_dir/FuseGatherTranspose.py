import torch
import triton
import triton.language as tl

# Pattern to match the index + reshape sequence (identical in both graphs!)
def pattern(table, indices):
    tmp = table[indices]
    tmp = tmp.view(197, 197, -1)
    tmp = tmp.permute(2, 0, 1)
    tmp = tmp.contiguous()
    tmp = tmp.unsqueeze(0)
    return tmp

def replacement_args(table, indices):
    return (table, indices)

# Optimized Triton kernel - 2D grid for better parallelism
@triton.jit
def gather_transpose_2d_v2(
    table_ptr,
    indices_ptr,
    output_ptr,
    num_indices,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (block_id, head_idx)
    block_id = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Calculate offsets for this block
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_indices
    
    # Load indices - coalesced read
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)
    
    # Load from table: table[idx, head_idx] - scattered read
    table_offs = idx * num_cols + head_idx
    val = tl.load(table_ptr + table_offs, mask=mask, other=0.0)
    
    # Store to output: output[head_idx, offs] - coalesced write
    out_offs = head_idx * num_indices + offs
    tl.store(output_ptr + out_offs, val, mask=mask)


@torch.fx.wrap
def fused_gather_transpose(table, indices):
    # table: [732, num_heads] (12 or 16)
    # indices: [38809] on CPU
    num_indices = indices.shape[0]  # 38809
    num_heads = table.shape[1]  # 12 or 16
    
    # Move indices to GPU with non-blocking transfer
    indices_gpu = indices.to(device=table.device, non_blocking=True)
    
    # Output shape: [1, num_heads, 197, 197]
    output = torch.empty((1, num_heads, 197, 197), dtype=table.dtype, device=table.device)
    
    BLOCK_SIZE = 2048
    num_blocks = (num_indices + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # 2D grid for full parallelism
    gather_transpose_2d_v2[(num_blocks, num_heads)](
        table_ptr=table,
        indices_ptr=indices_gpu,
        output_ptr=output,
        num_indices=num_indices,
        num_cols=num_heads,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_gather_transpose