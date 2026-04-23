import torch
import triton
import triton.language as tl

# Pattern matching function - simple index_select
def pattern(x, indices):
    return x.index_select(-2, indices)

def replacement_args(x, indices):
    return (x, indices)

@triton.jit
def triton_index_select_kernel(
    x_ptr,
    indices_ptr,
    out_ptr,
    x_row_stride: tl.int32,
    indices_ptr_add: tl.int32,
    out_row_stride: tl.int32,
    x_num_cols: tl.int32,
    num_indices: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one index
    pid = tl.program_id(0)
    
    # Bounds check
    if pid >= num_indices:
        return
    
    # Load the index value
    idx = tl.load(indices_ptr + pid * indices_ptr_add).to(tl.int32)
    
    # Calculate source and destination offsets
    x_base = idx * x_row_stride
    out_base = pid * out_row_stride
    
    # Process columns in blocks for better memory access
    for col_offset in range(0, x_num_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < x_num_cols
        
        # Load from source row
        x_ptrs = x_base + col_offsets
        x_vals = tl.load(x_ptr + x_ptrs, mask=mask, other=0.0)
        
        # Store to destination row
        out_ptrs = out_base + col_offsets
        tl.store(out_ptr + out_ptrs, x_vals, mask=mask)


@torch.fx.wrap
def triton_index_select_wrapper(x, indices):
    """
    Optimized index_select using Triton kernel.
    """
    num_indices = indices.numel()
    num_rows, num_cols = x.shape
    
    output = torch.empty((num_indices, num_cols), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_programs = triton.next_power_of_2(num_indices)
    num_programs = min(max(num_programs, 1), 4096)
    
    x_row_stride = x.stride(0)
    indices_ptr_add = indices.stride(0) if indices.dim() > 0 else 1
    out_row_stride = output.stride(0)
    
    triton_index_select_kernel[(num_programs,)](
        x,
        indices,
        output,
        x_row_stride,
        indices_ptr_add,
        out_row_stride,
        num_cols,
        num_indices,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return triton_index_select_wrapper