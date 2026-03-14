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

# Triton kernel for fused gather + transpose
@triton.jit
def gather_transpose_kernel(
    table_ptr,
    indices_ptr,
    output_ptr,
    num_indices,  # 38809 = 197*197
    num_cols,     # 12 or 16 (heads)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one head (column in output)
    head_idx = tl.program_id(0)
    
    # Process BLOCK_SIZE indices at a time
    for start in range(0, num_indices, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_indices
        
        # Load indices (convert from int64 to int32 for indexing)
        idx = tl.load(indices_ptr + offs, mask=mask, other=0)
        
        # Load from table: table[idx, head_idx]
        table_offs = idx * num_cols + head_idx
        val = tl.load(table_ptr + table_offs, mask=mask, other=0.0)
        
        # Store to output: output[head_idx, offs // 197, offs % 197]
        # Which in flattened form is: head_idx * 197 * 197 + offs
        out_offs = head_idx * num_indices + offs
        tl.store(output_ptr + out_offs, val, mask=mask)


@torch.fx.wrap
def fused_gather_transpose(table, indices):
    # table: [732, num_heads] (12 or 16)
    # indices: [38809] on CPU
    num_indices = indices.shape[0]  # 38809
    num_heads = table.shape[1]  # 12 or 16
    
    # Move indices to GPU
    indices_gpu = indices.to(table.device)
    
    # Output shape: [1, num_heads, 197, 197]
    output = torch.empty((1, num_heads, 197, 197), dtype=table.dtype, device=table.device)
    
    BLOCK_SIZE = 1024
    
    gather_transpose_kernel[(num_heads,)](
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