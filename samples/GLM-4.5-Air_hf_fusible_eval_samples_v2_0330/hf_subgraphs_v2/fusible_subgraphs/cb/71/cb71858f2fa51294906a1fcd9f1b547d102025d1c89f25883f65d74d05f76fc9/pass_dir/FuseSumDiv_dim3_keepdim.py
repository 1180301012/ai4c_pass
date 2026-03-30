import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_5 = x.sum(dim=3, keepdim=True)
    tmp_6 = x / tmp_5
    return tmp_6

def replacement_args(x):
    return (x,)

@triton.jit
def fused_sum_div_kernel(
    x_ptr,
    out_ptr,
    batch, channels, height, width,
):
    # Each program handles one row (batch * channel * height dim3)
    pid = tl.program_id(0)
    
    # Calculate which row we're processing
    total_rows = batch * channels * height
    if pid >= total_rows:
        return
    
    # Calculate coordinates
    row_idx = pid
    batch_idx = row_idx // (channels * height)
    channel_idx = (row_idx % (channels * height)) // height
    height_idx = row_idx % height
    
    # Calculate starting position for this row
    start_pos = batch_idx * channels * height * width + channel_idx * height * width + height_idx * width
    
    # Initialize row sum
    row_sum = 0.0
    
    # Sum all elements in the row (dimension 3)
    for i in range(width):
        element_pos = start_pos + i
        x_val = tl.load(x_ptr + element_pos)
        row_sum += x_val
    
    # Compute division for each element in the row
    for i in range(width):
        element_pos = start_pos + i
        x_val = tl.load(x_ptr + element_pos)
        out_val = x_val / row_sum
        tl.store(out_ptr + element_pos, out_val)

@torch.fx.wrap
def fused_sum_div(x):
    # Get input shape
    batch, channels, height, width = x.shape
    
    # Output shape is the same as input
    out_shape = (batch, channels, height, width)
    
    # Calculate total number of rows (one program per row)
    total_rows = batch * channels * height
    
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel - each program handles one entire row
    fused_sum_div_kernel[(total_rows,)](
        x_ptr=x,
        out_ptr=output,
        batch=batch, channels=channels, height=height, width=width
    )
    
    return output

def replacement_func():
    return fused_sum_div