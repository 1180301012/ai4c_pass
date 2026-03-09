import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation structure for division by 2.0
    tmp_0 = in_0
    tmp_1 = in_1 / 2.0
    tmp_2 = tmp_0.view(-1)
    return tmp_1, tmp_2

def replacement_args(in_0, in_1):
    # Extract both tensors and constants
    constant = 2.0
    reciprocal = 1.0 / constant  # This equals 0.5
    return (in_0, in_1, reciprocal)

@triton.jit
def combined_div_two_view_kernel(
    x_ptr,
    y_ptr,
    reciprocal,
    out1_ptr,
    out2_ptr,
    n_elements_x,
    n_elements_y,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Handle view operation on x tensor
    block_start_x = tl.program_id(0) * BLOCK_SIZE_X
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    mask_x = offsets_x < n_elements_x
    
    # Load x tensor for view operation (flatten)
    x = tl.load(x_ptr + offsets_x, mask=mask_x, other=0)
    # Store flattened result
    tl.store(out2_ptr + offsets_x, x, mask=mask_x)
    
    # Handle division operation on y tensor
    if tl.program_id(1) > 0:  # Only one program handles y for this example
        return
        
    block_start_y = tl.program_id(1) * BLOCK_SIZE_Y
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    mask_y = offsets_y < n_elements_y
    
    # Load y tensor
    y = tl.load(y_ptr + offsets_y, mask=mask_y, other=0.0)
    # Multiply by reciprocal (0.5) instead of dividing by 2.0
    out1 = y * reciprocal
    # Store division result
    tl.store(out1_ptr + offsets_y, out1, mask=mask_y)

@torch.fx.wrap
def optimized_combined_div_two_view(x, y, reciprocal):
    # Handle view operation on x
    if not x.is_contiguous():
        x = x.contiguous()
    
    N_x = x.numel()
    N_y = y.numel()
    
    BLOCK_SIZE_X = 1024
    BLOCK_SIZE_Y = 1024
    
    num_programs_x = (N_x + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    num_programs_y = (N_y + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Create output tensors
    out1 = torch.empty_like(y)  # Division result
    out2 = torch.empty(N_x, dtype=x.dtype, device=x.device)  # View result
    
    # Launch combined kernel
    grid_x = (num_programs_x,)
    grid_y = (1,)
    combined_div_two_view_kernel[(grid_x, grid_y)](
        x_ptr=x,
        y_ptr=y,
        reciprocal=reciprocal,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements_x=N_x,
        n_elements_y=N_y,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return out1, out2

def replacement_func():
    return optimized_combined_div_two_view