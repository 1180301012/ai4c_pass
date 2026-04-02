import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match ReLU + Dropout2d pattern where Dropout2d has training=False (inference mode)
    This pattern is common in inference where dropout is disabled but graph structure needs to be preserved
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace = True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return tmp_1, tmp_0


def replacement_args(in_0):
    """
    Extract arguments for the replacement - just the input tensor
    """
    return (in_0,)


@triton.jit
def relu_kernel_forward(
    x_ptr,
    y_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    High-performance Triton kernel for ReLU operation that writes output to two locations
    This eliminates the redundant Dropout2d operation when training=False by
    computing ReLU once and writing results to both output locations
    """
    # Calculate 3D program ID for 4D tensor [batch, channels, height, width]
    pid = tl.program_id(0)
    # Total elements per program (assuming BLOCK_SIZE divides total elements)
    total_elements = input_batch * input_channels * input_height * input_width
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate start element for this program
    start_element = pid * elements_per_program
    # Calculate memory offsets for this program
    offsets = start_element + tl.arange(0, min(elements_per_program, BLOCK_SIZE))
    
    # Create mask for out-of-bounds elements
    mask = offsets < total_elements
    
    # Calculate 4D indices from flat index
    batch_idx = offsets // (input_channels * input_height * input_width)
    remainder = offsets % (input_channels * input_height * input_width)
    channel_idx = remainder // (input_height * input_width)
    remainder = remainder % (input_height * input_width)
    height_idx = remainder // input_width
    width_idx = remainder % input_width
    
    # Calculate memory addresses
    x_addr = x_ptr + batch_idx * input_channels * input_height * input_width + \
             channel_idx * input_height * input_width + \
             height_idx * input_width + width_idx
    
    y_addr1 = y_ptr + batch_idx * input_channels * input_height * input_width + \
              channel_idx * input_height * input_width + \
              height_idx * input_width + width_idx
    
    y_addr2 = y_ptr + (pid + 1) * input_batch * input_channels * input_height * input_width + \
              batch_idx * input_channels * input_height * input_width + \
              channel_idx * input_height * input_width + \
              height_idx * input_width + width_idx
    
    # Load input data
    x = tl.load(x_addr, mask=mask, other=0.0)
    
    # Apply ReLU activation
    y = tl.maximum(x, 0.0)
    
    # Store results to both output locations (mimicking original tmp_1 and tmp_0)
    # Note: This simplified approach writes to contiguous memory for each output
    # For the actual pass, we'll use two separate kernels for clarity
    
    # Store ReLU result (will be used for both outputs)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def simple_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple and efficient 1D ReLU kernel for better performance
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_relu_dropout_inference(x_orig):
    """
    Optimized function that eliminates redundant Dropout2d operation
    during inference and returns ReLU result twice
    """
    # Create output tensors (both will contain the ReLU result)
    # tmp_1 equivalent ( Dropout2d output - should be same as ReLU when training=False)
    tmp_1 = torch.empty_like(x_orig)
    # tmp_0 equivalent (ReLU output)
    tmp_0 = torch.empty_like(x_orig)
    
    # If both outputs should be identical, we can compute ReLU once
    # and duplicate the result
    N = x_orig.numel()
    BLOCK_SIZE = 1024
    
    # Single kernel call to compute ReLU
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_relu_kernel[(num_programs,)](
        x_ptr=x_orig,
        out_ptr=tmp_1,  # Write to first output
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Copy first output to second output (since Dropout2d during inference is identity)
    tmp_0.copy_(tmp_1)  # This is more efficient than recomputing
    
    return tmp_1, tmp_0


def replacement_func():
    """
    Returns the optimized function that replaces the original ReLU+Dropout2d pattern
    """
    return optimized_relu_dropout_inference