import torch
import triton
import triton.language as tl

def pattern(mesh_input1, mesh_input2):
    """
    Optimized meshgrid pattern - improved for better performance.
    """
    mesh_result = torch.meshgrid(mesh_input1, mesh_input2, indexing='ij')
    return mesh_result

def replacement_args(mesh_input1, mesh_input2):
    return (mesh_input1, mesh_input2)

@triton.jit
def optimized_meshgrid_kernel(
    input1_ptr, input2_ptr,
    out1_ptr, out2_ptr,
    input1_size, input2_size, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Use vectorized loads for better performance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Optimized coordinate calculation
    rows = offsets // input2_size
    cols = offsets % input2_size
    
    # Load data with optimized masking
    val1 = tl.load(input1_ptr + rows, mask=(rows < input1_size), other=0.0)
    val2 = tl.load(input2_ptr + cols, mask=(cols < input2_size), other=0.0)
    
    # Store results with vectorized stores
    tl.store(out1_ptr + offsets, val1, mask=mask)
    tl.store(out2_ptr + offsets, val2, mask=mask)

@torch.fx.wrap
def optimized_highperf_meshgrid(input1, input2):
    """High-performance meshgrid with optimized memory access"""
    input1_size = input1.numel()
    input2_size = input2.numel()
    total_elements = input1_size * input2_size
    
    # Pre-allocate output tensors with better alignment
    out1 = torch.empty(total_elements, dtype=input1.dtype, device=input1.device)
    out2 = torch.empty(total_elements, dtype=input2.dtype, device=input2.device)
    
    # Optimal block size for larger tensors
    BLOCK_SIZE = 512 if total_elements > 10000 else 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_meshgrid_kernel[(num_programs,)](
        input1_ptr=input1,
        input2_ptr=input2,
        out1_ptr=out1,
        out2_ptr=out2,
        input1_size=input1_size,
        input2_size=input2_size,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1.reshape(input1_size, input2_size), out2.reshape(input1_size, input2_size)

def replacement_func():
    return optimized_highperf_meshgrid