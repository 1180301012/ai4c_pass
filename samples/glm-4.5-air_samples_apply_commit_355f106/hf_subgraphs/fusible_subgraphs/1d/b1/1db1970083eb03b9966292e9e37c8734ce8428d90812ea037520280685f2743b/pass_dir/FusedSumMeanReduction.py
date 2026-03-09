import torch
import triton
import triton.language as tl

# Pattern matching function - matches the sequential sum then mean operations
def pattern(x):
    tmp_0 = x.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_1,)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Simple working fused kernel using Triton - avoid arange issues
@triton.jit
def fused_sum_mean_kernel(x_ptr, out_ptr, n_elements, height2):
    # Each program handles one h2 element
    h2 = tl.program_id(0)
    
    # Initialize accumulator for this h2 position
    # This is a simple kernel that processes one element at a time
    total_sum = 0.0
    element_count = 0
    
    # Simple loop - for now, just process a fixed number of relevant offsets
    # Based on the reduction pattern: sum over channels (dim=1), mean over height1, width1 (dims 2,3)
    # So for each h2, we need to access all c, h1, w1 combinations
    BLOCK_SIZE = 64
    for block_start in range(0, n_elements // height2, BLOCK_SIZE):
        # Process a block of elements for this h2
        for i in range(BLOCK_SIZE):
            flat_idx = block_start * height2 + i * height2 + h2
            if flat_idx < n_elements:
                val = tl.load(x_ptr + flat_idx)
                total_sum += val
                element_count += 1
    
    # Compute mean
    if element_count > 0:
        result = total_sum / element_count
    else:
        result = 0.0
    
    # Store result
    tl.store(out_ptr + h2, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_sum_mean(x):
    # Get input shape and compute output shape
    input_shape = x.shape
    batch, channels, height1, width1, height2 = input_shape
    
    # Output will have shape [1, 1, 1, 1, height2] due to keepdim=True
    output_shape = [1, 1, 1, 1, height2]
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Flatten output to 1D for simpler kernel - shape [height2]
    output_flat = output.squeeze()  # [height2]
    
    # Grid configuration: we launch height2 number of programs, one per output element
    # Each program handles one h2 dimension element with vectorized processing
    BLOCK_SIZE = 1024  # Number of elements each thread program processes
    
    # Total number of programs needed (one per height2 element)
    num_programs = height2
    
    # Calculate total elements in the input tensor
    n_elements = channels * height1 * width1 * height2
    
    # Launch the kernel - pass raw pointers
    fused_sum_mean_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=output_flat,
        n_elements=n_elements,
        height2=height2,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_sum_mean