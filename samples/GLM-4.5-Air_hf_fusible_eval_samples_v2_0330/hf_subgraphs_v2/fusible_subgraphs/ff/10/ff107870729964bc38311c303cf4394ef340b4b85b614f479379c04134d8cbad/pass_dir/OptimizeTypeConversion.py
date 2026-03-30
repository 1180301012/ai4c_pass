import torch
import triton
import triton.language as tl

def pattern(input_tensor, target_dtype):
    """Match type conversion operation"""
    # Type conversion to match the model pattern
    converted_tensor = input_tensor.to(target_dtype)
    return converted_tensor

def replacement_args(input_tensor, target_dtype):
    return (input_tensor, target_dtype)

@triton.jit
def type_conversion_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    """Optimized type conversion kernel"""
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Ensure bounds
    if pid_n >= N or pid_c >= C:
        return
    
    # Process spatial dimensions for this batch and channel
    offset_base = pid_n * (C * H * W) + pid_c * (H * W)
    
    for i in range(H * W):
        offset = offset_base + i
        
        # Load input value as float32 for precision
        x = tl.load(input_ptr + offset, dtype=tl.float32)
        
        # Convert to float16 (both float16 and bfloat16 supported)
        # Triton handles precision conversion automatically
        output_value = x
        
        # Store result - dtype will be handled by output tensor type
        tl.store(output_ptr + offset, output_value)

@triton.jit
def bfloat16_conversion_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    """BFloat16 optimized conversion kernel"""
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Ensure bounds
    if pid_n >= N or pid_c >= C:
        return
    
    # Process spatial dimensions for this batch and channel
    offset_base = pid_n * (C * H * W) + pid_c * (H * W)
    
    for i in range(H * W):
        offset = offset_base + i
        
        # Load input value
        x = tl.load(input_ptr + offset, dtype=tl.float32)
        
        # For bfloat16, we can use round-to-nearest-even
        # Convert to bfloat16 by rounding and then back to float32
        x_rounded = tl.round(x)
        bfloat16_val = x_rounded
        
        # Store result
        tl.store(output_ptr + offset, bfloat16_val)

@torch.fx.wrap
def optimized_type_conversion(input_tensor, target_dtype):
    """Optimized type conversion function"""
    N, C, H, W = input_tensor.shape
    
    # Create output tensor with target dtype
    output = torch.empty((N, C, H, W), dtype=target_dtype, device=input_tensor.device)
    
    # Choose kernel based on target dtype
    if target_dtype == torch.float16:
        kernel = type_conversion_kernel
    elif target_dtype == torch.bfloat16:
        kernel = bfloat16_conversion_kernel
    else:
        # For other dtypes, fall back to PyTorch implementation
        return input_tensor.to(target_dtype)
    
    # Launch Triton kernel
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 64
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    kernel[(grid_n, grid_c)](
        input_tensor,
        output,
        N, C, H, W,
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
    )
    
    return output

def optimized_noop_conversion(input_tensor, target_dtype):
    """Optimized conversion that may be a no-op if dtype already matches"""
    if input_tensor.dtype == target_dtype:
        # No conversion needed, return original tensor
        return input_tensor
    else:
        # Perform optimized conversion
        return optimized_type_conversion(input_tensor, target_dtype)

def replacement_func():
    return optimized_noop_conversion