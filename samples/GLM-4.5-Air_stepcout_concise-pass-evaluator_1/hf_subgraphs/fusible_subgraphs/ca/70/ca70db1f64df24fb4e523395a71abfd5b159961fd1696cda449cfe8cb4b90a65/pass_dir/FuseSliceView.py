import torch
import triton
import triton.language as tl

# Pattern matching function - slices first 256 columns and reshapes
def pattern(input_tensor):
    """
    Pattern: slice first 256 columns then reshape to (..., 256)
    Corresponds to:
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    """
    sliced = input_tensor[slice(None, None, None), slice(None, 256, None)]
    reshaped = sliced.view(-1, 256)
    return reshaped

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for fused slice+reshape operation
@triton.jit
def fused_slice_kernel_first_256(
    input_ptr,
    output_ptr,
    input_size_0,
    input_size_1,
    output_size_0,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row of output
    row_idx = tl.program_id(0)
    
    # Load first 256 elements from the row
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < 256  # Always take first 256 columns
    
    ptr = input_ptr + row_idx * input_size_1
    out_row = tl.load(ptr + offsets, mask=mask, other=0.0)
    
    # Store directly to output
    out_ptr = output_ptr + row_idx * 256
    tl.store(out_ptr + offsets, out_row, mask=mask)

@torch.fx.wrap
def fused_slice_first_256(input_tensor):
    # Input shape: [M, 512], output shape: [M, 256]
    M, N_in = input_tensor.shape
    
    assert N_in >= 256, f"Input second dim must be >= 256, got {N_in}"
    
    output_shape = (M, 256)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - each program handles one complete row
    num_programs = M
    
    fused_slice_kernel_first_256[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_size_0=M,
        input_size_1=N_in,
        output_size_0=M,
        BLOCK_SIZE_N=256
    )
    
    return output

# Pattern matching function - slices last 256 columns and reshapes
def pattern_second(input_tensor):
    """
    Pattern: slice last 256 columns then reshape to (..., 256)
    Corresponds to:
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    """
    sliced = input_tensor[slice(None, None, None), slice(-256, None, None)]
    reshaped = sliced.view(-1, 256)
    return reshaped

# Argument extraction function for second pattern
def replacement_args_second(input_tensor):
    return (input_tensor,)

# Triton kernel for second fused slice+reshape operation (last 256 columns)
@triton.jit
def fused_slice_kernel_last_256(
    input_ptr,
    output_ptr,
    input_size_0,
    input_size_1,
    output_size_0,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row of output
    row_idx = tl.program_id(0)
    
    # Load last 256 elements from the row
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < 256  # Always take last 256 columns
    
    ptr = input_ptr + row_idx * input_size_1 + (input_size_1 - 256)
    out_row = tl.load(ptr + offsets, mask=mask, other=0.0)
    
    # Store directly to output
    out_ptr = output_ptr + row_idx * 256
    tl.store(out_ptr + offsets, out_row, mask=mask)

@torch.fx.wrap
def fused_slice_last_256(input_tensor):
    # Input shape: [M, 512], output shape: [M, 256]
    M, N_in = input_tensor.shape
    
    assert N_in >= 256, f"Input second dim must be >= 256, got {N_in}"
    
    output_shape = (M, 256)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - each program handles one complete row
    num_programs = M
    
    fused_slice_kernel_last_256[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_size_0=M,
        input_size_1=N_in,
        output_size_0=M,
        BLOCK_SIZE_N=256
    )
    
    return output

# Replacement function (returns both kernel wrappers)
def replacement_func():
    return fused_slice_first_256

# Additional replacement function for second pattern
def replacement_func_second():
    return fused_slice_last_256