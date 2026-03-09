import torch
import triton
import triton.language as tl

@triton.jit
def fused_max_sub_softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row in one batch
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    # Pointer arithmetic for current row
    x_batch_ptr = x_ptr + batch_idx * height * width
    x_row_ptr = x_batch_ptr + row_idx * width
    
    # Load current row
    row_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = row_offsets < width
    x_row = tl.load(x_row_ptr + row_offsets, mask=mask, other=-float('inf'))
    
    # Find max in the row
    max_val = tl.max(x_row, axis=0)
    
    # Subtract max from original values (numerical stability for softmax)
    exp_x = tl.exp(x_row - max_val)
    
    # Compute softmax sum
    exp_sum = tl.sum(exp_x * mask)
    
    # Compute softmax values
    softmax_row = exp_x / (exp_sum + 1e-8)
    
    # Store result
    out_batch_ptr = out_ptr + batch_idx * height * width
    out_row_ptr = out_batch_ptr + row_idx * width
    tl.store(out_row_ptr + row_offsets, softmax_row, mask=mask)

@torch.fx.wrap
def fused_max_sub_softmax(x):
    batch_size, height, width = x.shape
    
    # Calculate grid size
    if width <= 1024:
        BLOCK_SIZE_N = 1024
    else:
        BLOCK_SIZE_N = 2048
    
    # Ensure block size doesn't exceed width
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, width)
    
    # Grid size: (batch_size, height, 1)
    grid = (batch_size, height, 1)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_max_sub_softmax_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        height=height,
        width=width,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

@torch.fx.wrap  
def optimized_view(x, batch_size, channels):
    return x.view(batch_size, channels, -1)

@torch.fx.wrap
def optimized_forward(x, y):
    # Apply fused max-sub-softmax to first input
    result_1 = fused_max_sub_softmax(x)
    
    # Apply optimized view to second input
    result_2 = optimized_view(y, x.shape[0], x.shape[1])
    
    return (result_1, result_2)

def pattern(self, x, y):
    # Very simple pattern to test - just match max and softmax on first tensor
    max_val = torch.max(x, -1, keepdim=True)[0]
    result = torch.nn.functional.softmax(max_val - x, dim=-1)
    return result

def replacement_args(self, x, y):
    return (x, y)

def replacement_func():
    def simple_fused_max_softmax(x, y):
        # Use only x for the fused operation (y is ignored for this pattern)
        batch_size, height, width = x.shape
        
        # Calculate grid size
        if width <= 1024:
            BLOCK_SIZE_N = 1024
        else:
            BLOCK_SIZE_N = 2048
        
        # Ensure block size doesn't exceed width
        BLOCK_SIZE_N = min(BLOCK_SIZE_N, width)
        
        # Grid size: (batch_size, height, 1)
        grid = (batch_size, height, 1)
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Launch simplified kernel
        fused_max_sub_softmax_kernel[grid](
            x_ptr=x,
            out_ptr=out,
            batch_size=batch_size,
            height=height,
            width=width,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        
        return out
    return simple_fused_max_softmax