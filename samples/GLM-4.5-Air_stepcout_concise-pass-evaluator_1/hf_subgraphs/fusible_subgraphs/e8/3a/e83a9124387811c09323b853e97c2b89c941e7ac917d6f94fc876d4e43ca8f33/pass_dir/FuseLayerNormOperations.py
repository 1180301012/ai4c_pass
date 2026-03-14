import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_kernel(
    # Input tensors
    in_2_ptr,          # Layer output tensor  
    in_3_ptr,          # Layer output tensor
    in_1_ptr,          # Weight tensor (broadcasted)
    in_0_ptr,          # Bias tensor (broadcasted)
    
    # Output tensors  
    out_full_ptr,      # Full output
    
    # Tensor shapes
    dim0_size,         # Size of first dimension
    dim1_size,         # Size of second dimension  
    dim2_size,         # Size of third dimension
    param_size,        # Size of parameter tensors
    
    # Strides
    in_2_stride0,
    in_2_stride1, 
    in_2_stride2,
    in_3_stride0,
    in_3_stride1,
    in_3_stride2,
    in_1_stride0,
    in_0_stride0,
    out_full_stride0,
    out_full_stride1,
    out_full_stride2,
    
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs for parallel processing
    pid_dim0 = tl.program_id(0)  # Process different dim0 elements
    pid_block = tl.program_id(1)  # Process different blocks in flattened space
    
    # Only process valid dim0 elements
    if pid_dim0 >= dim0_size:
        return
    
    # Process elements in flattened dim1-dim2 space
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (dim1_size * dim2_size)
    
    # Convert linear offset to 3D coordinates
    dim2_coords = offsets % dim2_size
    dim1_coords = (offsets // dim2_size) % dim1_size
    
    # Calculate memory addresses for all tensors
    in_2_addr = (in_2_ptr + 
               pid_dim0 * in_2_stride0 + 
               dim1_coords * in_2_stride1 + 
               dim2_coords * in_2_stride2)
    
    in_3_addr = (in_3_ptr + 
               pid_dim0 * in_3_stride0 + 
               dim1_coords * in_3_stride1 + 
               dim2_coords * in_3_stride2)
    
    # Load input values
    in_2_val = tl.load(in_2_addr, mask=mask, other=0.0)
    in_3_val = tl.load(in_3_addr, mask=mask, other=0.0)
    
    # Load broadcasted parameters (same for all positions in this slice)
    param_offset = dim2_coords % param_size
    
    in_1_addr = in_1_ptr + param_offset * in_1_stride0
    in_0_addr = in_0_ptr + param_offset * in_0_stride0
    
    in_1_val = tl.load(in_1_addr, mask=(param_offset < param_size), other=1.0)
    in_0_val = tl.load(in_0_addr, mask=(param_offset < param_size), other=0.0)
    
    # Fused computation: ((in_3 + in_2) * in_1) + in_0
    result = (in_3_val + in_2_val) * in_1_val + in_0_val
    
    # Store to full output
    out_full_addr = (out_full_ptr + 
                   pid_dim0 * out_full_stride0 + 
                   dim1_coords * out_full_stride1 + 
                   dim2_coords * out_full_stride2)
    tl.store(out_full_addr, result, mask=mask)

@torch.fx.wrap
def fused_layer_norm_operation(in_0, in_1, in_2, in_3):
    """
    Optimized fused operation: ((in_3 + in_2) * in_1) + in_0
    with automatic broadcasting and slicing the first element along dim 0.
    """
    # Get input shapes for proper handling
    param_size = in_0.numel()  # Size of bias/weight parameters
    
    # Create output tensor
    out_full = torch.empty(in_2.shape, dtype=torch.float32, device=in_2.device)
    
    # Calculate tensor strides
    in_2_stride0, in_2_stride1, in_2_stride2 = in_2.stride()
    in_3_stride0, in_3_stride1, in_3_stride2 = in_3.stride()
    in_1_stride0 = in_1.stride(0) if in_1.dim() == 1 else 0
    in_0_stride0 = in_0.stride(0) if in_0.dim() == 1 else 0
    out_full_stride0, out_full_stride1, out_full_stride2 = out_full.stride()
    
    # Extract tensor dimensions
    dim0_size, dim1_size, dim2_size = in_2.shape
    
    # Choose optimal block size for flattened processing
    BLOCK_SIZE = 1024  # Process 1024 elements per program
    
    # Calculate grid dimensions for 2D parallelism
    total_elements = dim1_size * dim2_size
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_dim0 = dim0_size
    
    # Launch optimized Triton kernel with 2D grid
    fused_layer_norm_kernel[(grid_dim0, num_blocks)](
        # Input tensors
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        
        # Output tensors
        out_full_ptr=out_full,
        
        # Tensor shapes
        dim0_size=dim0_size,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        param_size=param_size,
        
        # Strides
        in_2_stride0=in_2_stride0,
        in_2_stride1=in_2_stride1,
        in_2_stride2=in_2_stride2,
        in_3_stride0=in_3_stride0,
        in_3_stride1=in_3_stride1,
        in_3_stride2=in_3_stride2,
        in_1_stride0=in_1_stride0,
        in_0_stride0=in_0_stride0,
        out_full_stride0=out_full_stride0,
        out_full_stride1=out_full_stride1,
        out_full_stride2=out_full_stride2,
        
        # Block size
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_full

# Pattern matching function - must match the exact computation in model.py
def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: ((in_3 + in_2) * in_1) + in_0
    This matches the computation that produces tmp_4
    """
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Replacement function that returns the optimized kernel
def replacement_func():
    return fused_layer_norm_operation