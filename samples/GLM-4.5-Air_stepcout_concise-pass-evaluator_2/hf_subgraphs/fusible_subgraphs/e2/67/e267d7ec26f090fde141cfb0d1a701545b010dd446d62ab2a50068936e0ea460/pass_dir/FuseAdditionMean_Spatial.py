import torch
import triton
import triton.language as tl

# Pattern matching function - matches addition operations followed by mean over spatial dimensions
def pattern(a, b, c=None):
    """
    Matches the computation pattern exactly as seen in the model:
    - Multiple tensor additions (2 or 3 inputs)
    - Mean operation over spatial dimensions (2, 3) with keepdim=True
    """
    if c is not None:
        # 3-input addition pattern: c + b, then + a (matching the model structure)
        tmp_0 = b + c
        tmp_0 += a
        tmp_1 = tmp_0
        tmp_2 = tmp_1.mean((2, 3), keepdim=True)
        return tmp_1, tmp_2
    else:
        # 2-input addition pattern: b + a (matching the model structure)  
        tmp_0 = b + a
        tmp_1 = tmp_0
        tmp_2 = tmp_1.mean((2, 3), keepdim=True)
        return tmp_1, tmp_2

# Argument extraction function
def replacement_args(a, b, c=None):
    """Extract arguments for the replacement kernel"""
    return (a, b, c)

# Kernel for computing spatial mean (separate due to different parallelization strategy)
@triton.jit
def spatial_mean_kernel(
    sum_ptr,
    mean_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Compute mean over spatial dimensions (2, 3) with keepdim=True"""
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel
    
    # Compute spatial mean for this batch and channel
    spatial_sum = 0.0
    hw = height * width
    
    # Load all spatial elements for this batch+channel and sum them
    for hw_idx in range(hw):
        spatial_sum += tl.load(sum_ptr + (pid_b * channels * height * width) + 
                              (pid_c * height * width) + hw_idx)
    
    # Compute mean
    spatial_mean = spatial_sum / hw
    
    # Store result (mean has shape [batch, channels, 1, 1])
    mean_offset = pid_b * channels + pid_c
    tl.store(mean_ptr + mean_offset, spatial_mean)

# Triton kernel for fused addition computation
@triton.jit
def fused_addition_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sum_ptr,
    batch_size,
    channels,
    height,
    width,
    input_count: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute sum of 2-3 input tensors"""
    n_elements = batch_size * channels * height * width
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and sum inputs
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load first input
    a_val = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    sum_val = a_val
    
    # Load and add second input
    b_val = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    sum_val += b_val
    
    # Load and add third input if present
    if input_count == 3:
        c_val = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        sum_val += c_val
    
    # Store the sum result
    tl.store(sum_ptr + offsets, sum_val, mask=mask)

# Wrapper function for the fused operation
@torch.fx.wrap
def fused_addition_mean_op(a, b, c=None):
    """
    Perform fused addition and mean computation using Triton
    """
    inputs = [a, b, c] if c is not None else [a, b]
    
    # Determine input count and handle constants
    input_count = 3 if c is not None else 2
    actual_inputs = []
    
    for i, input_val in enumerate(inputs):
        if isinstance(input_val, (int, float)):
            # Convert constant to tensor (use first non-constant tensor as reference)
            ref_tensor = next((x for x in inputs if isinstance(x, torch.Tensor)), None)
            if ref_tensor is None:
                raise ValueError("No reference tensor found for constant conversion")
            input_val = torch.full_like(ref_tensor, input_val)
        actual_inputs.append(input_val)
    
    # Get shapes
    shape = actual_inputs[0].shape
    batch_size, channels, height, width = shape
    n_elements = actual_inputs[0].numel()
    
    # Output tensors
    output_sum = torch.empty_like(actual_inputs[0])
    output_mean = torch.empty((batch_size, channels, 1, 1), dtype=actual_inputs[0].dtype, device=actual_inputs[0].device)
    
    # Launch addition kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare kernel arguments based on input count
    if input_count == 3:
        fused_addition_kernel[num_programs,](
            actual_inputs[0],
            actual_inputs[1], 
            actual_inputs[2],
            output_sum,
            batch_size,
            channels,
            height,
            width,
            3,
            BLOCK_SIZE,
        )
    else:
        fused_addition_kernel[num_programs,](
            actual_inputs[0],
            actual_inputs[1], 
            None,  # No third input
            output_sum,
            batch_size,
            channels,
            height,
            width,
            2,
            BLOCK_SIZE,
        )
    
    # Launch spatial mean kernel 
    # Parallelize over batch and channel dimensions
    spatial_mean_kernel[(batch_size, channels),](
        output_sum,
        output_mean,
        batch_size,
        channels,
        height,
        width,
        32,  # Block size for mean computation
    )
    
    return output_sum, output_mean

# Replacement function (no arguments, returns function reference)
def replacement_func():
    """Returns the fused operation function"""
    return fused_addition_mean_op