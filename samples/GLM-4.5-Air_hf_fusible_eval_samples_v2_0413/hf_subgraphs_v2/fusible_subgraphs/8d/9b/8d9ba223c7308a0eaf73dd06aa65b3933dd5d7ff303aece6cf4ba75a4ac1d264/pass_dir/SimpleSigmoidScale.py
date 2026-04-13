import torch
import triton
import triton.language as tl

# Pattern matching function for a simple sigmoid + scaling operation
def pattern(input_tensor):
    """
    Match the pattern: sigmoid followed by multiplication
    This appears in all target computations:
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    """
    tmp_9 = torch.sigmoid(input_tensor)
    tmp_10 = 16 * tmp_9
    return tmp_10

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel using Triton for bfloat16 data type
@triton.jit
def sigmoid_scale_bf16_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with optimized memory access
    x_bf16 = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to fp32 for computation (optimized conversion)
    x_fp32 = x_bf16.to(tl.float32)
    
    # Optimized sigmoid + scale computation with clipping
    # Use standard sigmoid formula for better numerical stability
    neg_x = -x_fp32
    # Clip to reasonable range for numerical stability
    clipped_neg_x = tl.maximum(neg_x, -50.0)
    exp_x = tl.exp(clipped_neg_x)
    result_fp32 = 1.0 / (1.0 + exp_x)
    result_fp32 = result_fp32 * scale
    
    # Convert back to bfloat16 and store
    result_bf16 = result_fp32.to(tl.bfloat16)
    tl.store(output_ptr + offsets, result_bf16, mask=mask)

# Optimized kernel using Triton for float16 data type
@triton.jit
def sigmoid_scale_fp16_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with optimized memory access
    x_fp16 = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to fp32 for computation (optimized conversion)
    x_fp32 = x_fp16.to(tl.float32)
    
    # Optimized sigmoid + scale computation with clipping
    # Use standard sigmoid formula for better numerical stability
    neg_x = -x_fp32
    # Clip to reasonable range for numerical stability
    clipped_neg_x = tl.maximum(neg_x, -50.0)
    exp_x = tl.exp(clipped_neg_x)
    result_fp32 = 1.0 / (1.0 + exp_x)
    result_fp32 = result_fp32 * scale
    
    # Convert back to float16 and store
    result_fp16 = result_fp32.to(tl.float16)
    tl.store(output_ptr + offsets, result_fp16, mask=mask)

@torch.fx.wrap
def simple_optimized_sigmoid_scale(input_tensor):
    """
    Optimized version of sigmoid + scaling operations
    """
    n_elements = input_tensor.numel()
    
    # Create output tensor with correct shape and data type
    output = torch.empty_like(input_tensor)
    
    # Choose optimal block size based on tensor size and data type
    if input_tensor.dtype == torch.bfloat16:
        # bfloat16 prefers larger blocks due to hardware optimization
        if n_elements < 5000:
            BLOCK_SIZE = 1024
        elif n_elements < 50000:
            BLOCK_SIZE = 2048
        else:
            BLOCK_SIZE = 4096
    else:  # float16
        # float16 can handle smaller blocks efficiently
        if n_elements < 5000:
            BLOCK_SIZE = 512
        elif n_elements < 50000:
            BLOCK_SIZE = 1024
        else:
            BLOCK_SIZE = 2048
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use appropriate kernel based on data type
    if input_tensor.dtype == torch.bfloat16:
        sigmoid_scale_bf16_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            n_elements=n_elements,
            scale=16.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif input_tensor.dtype == torch.float16:
        sigmoid_scale_fp16_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            n_elements=n_elements,
            scale=16.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For fp32, use a simpler highly optimized kernel
        @triton.jit
        def sigmoid_scale_fp32_kernel(
            input_ptr, output_ptr, n_elements, scale: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            block_start = tl.program_id(0) * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            neg_x = -x
            # Clip to reasonable range for numerical stability
            clipped_neg_x = tl.maximum(neg_x, -50.0)
            exp_x = tl.exp(clipped_neg_x)
            result = 1.0 / (1.0 + exp_x)
            result = result * scale
            tl.store(output_ptr + offsets, result, mask=mask)
        
        sigmoid_scale_fp32_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            n_elements=n_elements,
            scale=16.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

# Replacement function
def replacement_func():
    return simple_optimized_sigmoid_scale