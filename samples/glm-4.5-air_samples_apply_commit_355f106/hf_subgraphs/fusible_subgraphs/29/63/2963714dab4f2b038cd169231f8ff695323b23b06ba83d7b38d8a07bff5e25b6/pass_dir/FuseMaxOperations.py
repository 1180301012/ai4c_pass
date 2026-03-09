import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Original pattern: two consecutive max operations
    tmp_8 = x.max(0, keepdim=False)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_9.max(-1, keepdim=True)
    tmp_11 = tmp_10[0]
    return tmp_11

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_max_kernel_2d(
    x_ptr,
    out_ptr,
    rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Calculate offsets for this program
    offset = row_idx * cols + col_idx
    
    # For max along dim 0 then dim -1, we can optimize:
    # First max reduces along dim 0, second along dim -1
    # Since we know the input dimensions, we can fuse this
    
    if row_idx == 0:
        # For first row (after max along dim 0), find max along columns
        max_val = tl.float32(-float('inf'))
        
        # Load the entire column [row_idx, col_idx] for all rows
        for i in range(rows):
            elem_offset = i * cols + col_idx
            elem = tl.load(x_ptr + elem_offset, other=0.0)
            if elem > max_val or tl.math.isinf(max_val):
                max_val = elem if not tl.math.isnan(elem) else max_val
        
        # Store the result - this is max along both dimensions
        tl.store(out_ptr + col_idx, max_val, mask=col_idx < cols)

@torch.fx.wrap
def fused_max_operations(x, y):
    # Handle different input dimensions
    if x.dim() == 2:
        rows, cols = x.shape
        BLOCK_SIZE = 1024
        
        # For 2D input: max along dim 0 then dim -1
        # This is equivalent to finding the global maximum
        out = torch.empty((cols,), dtype=x.dtype, device=x.device)
        
        if rows > 0 and cols > 0:
            fused_max_kernel_2d[(cols, 1)](
                x_ptr=x,
                out_ptr=out,
                rows=rows,
                cols=cols,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # Handle edge cases
            if cols > 0:
                out[:] = float('-inf')
            # Empty case is handled by the default initialization
            
    elif x.dim() == 3:
        # Handle 3D case if needed (from the expansion in original)
        batch_size, rows, cols = x.shape
        out = torch.empty((batch_size, cols), dtype=x.dtype, device=x.device)
        
        # For max along dim 0 (across expanded dimension), then max along last dim
        # This effectively gives us max across the expanded dimension and last dim
        if batch_size > 0:
            # First, compute max along dim 0
            max_dim0 = x.max(dim=0)[0]  # This reduces the batch dimension
            
            # Then compute max along last dim
            out = max_dim0.max(dim=-1, keepdim=False)[0]
        
    else:
        # Fallback to original implementation for other dimensions
        tmp_8 = x.max(0, keepdim=False)
        tmp_9 = tmp_8[0]
        tmp_10 = tmp_9.max(-1, keepdim=True)
        out = tmp_10[0]
    
    return out

def replacement_func():
    return fused_max_operations