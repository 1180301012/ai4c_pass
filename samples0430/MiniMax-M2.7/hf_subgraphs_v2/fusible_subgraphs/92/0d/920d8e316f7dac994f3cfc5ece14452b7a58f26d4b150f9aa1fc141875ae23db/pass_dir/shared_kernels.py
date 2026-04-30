"""
Shared Triton kernels for the arange -> view -> repeat fusion optimization.
These kernels are imported by all pass files to ensure they share the same replacement_func.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_arange_view_repeat_kernel_impl(
    output_ptr,
    n_elements: tl.constexpr,
    repeat_dim0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. arange(0, n_elements) -> creates [0, 1, 2, ..., n_elements-1]
    2. view(1, -1) -> reshapes to (1, n_elements)
    3. repeat(repeat_dim0, 1) -> repeats repeat_dim0 times along dim 0
    
    Output shape: (repeat_dim0, n_elements)
    """
    output_row_size = n_elements
    
    # Each program processes one row of the output
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate which input element maps to each output position
    # For row at index row_idx, we want input element: row_idx % repeat_dim0
    # For column at index col, we want input element: col
    input_row = row_idx % repeat_dim0
    input_col = col_offsets
    
    # Input index (view flattens it to 1D)
    input_idx = input_row * output_row_size + input_col
    
    # Mask for bounds checking
    mask = input_idx < n_elements
    
    # Load the value from arange (which is just the index itself)
    val = input_idx.to(tl.float32)
    
    # Store to output
    output_idx = row_idx * output_row_size + col_offsets
    output_mask = output_idx < (repeat_dim0 * n_elements)
    tl.store(output_ptr + output_idx, val, mask=output_mask)


@torch.fx.wrap
def fused_arange_view_repeat_dispatcher(n_elements, repeat_dim0, dtype_name, device_str, route=""):
    """
    Unified dispatcher that routes to the appropriate kernel based on the route string.
    
    Args:
        n_elements: The end value for arange (e.g., 128 or 1000)
        repeat_dim0: First dimension of repeat (typically 2)
        dtype_name: String name of the dtype ('float32', 'float16', 'bfloat16')
        device_str: String representation of device
        route: Route string to identify which pass is calling
    
    Returns:
        Tensor of shape (repeat_dim0, n_elements) with values [0, 1, ..., n_elements-1] repeated
    """
    BLOCK_SIZE = 1024
    num_rows = repeat_dim0
    
    # Parse dtype from name
    if dtype_name == 'float32':
        dtype = torch.float32
    elif dtype_name == 'float16':
        dtype = torch.float16
    elif dtype_name == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    # Parse device
    if device_str == "cuda" or device_str == "device(type='cuda')":
        device = torch.device('cuda')
    else:
        device = torch.device(device_str)
    
    # Allocate output tensor
    output = torch.empty((num_rows, n_elements), dtype=dtype, device=device)
    
    # Launch kernel with one program per output row
    fused_arange_view_repeat_kernel_impl[(num_rows,)](
        output_ptr=output,
        n_elements=n_elements,
        repeat_dim0=repeat_dim0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output