import torch
import triton
import triton.language as tl

# Pattern matching function for linear + view + indexing optimization
def pattern(linear_out, index_tensor, weight_tensor):
    """
    Match the pattern: linear_out.view(-1, weight.shape[0])[index_tensor.view(-1)]
    This appears in all target computations:
    linear = torch.nn.functional.linear(in_4, in_1, None)
    tmp_3 = linear.view(-1, 12)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    """
    # Extract output features from weight tensor (second dimension)
    output_features = weight_tensor.shape[0]
    
    # Match the exact operations from the target computation
    tmp_3 = linear_out.view(-1, output_features)
    tmp_4 = index_tensor.view(-1)
    tmp_5 = tmp_3[tmp_4]
    return tmp_5

# Argument extraction function
def replacement_args(input_tensor, weight_tensor, index_tensor):
    return (input_tensor, index_tensor, weight_tensor)

# Optimized kernel using Triton - specializing for linear + indexing
@triton.jit
def linear_view_index_kernel(
    input_ptr,
    weight_ptr,
    index_ptr,
    output_ptr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs
    pid = tl.program_id(0)
    
    # Each program handles one output row (reduction over input features)
    row_start = pid * output_features
    row_offset = tl.arange(0, output_features)
    mask = row_offset < output_features
    
    # Load index for this output row
    index_val = tl.load(index_ptr + pid, mask=pid < batch_size, other=0)
    
    # Accumulate linear operation result
    acc = tl.zeros(output_features, dtype=tl.float32)
    
    # Block loop for input features to improve memory locality
    for k in range(0, input_features, BLOCK_SIZE):
        block_end = min(k + BLOCK_SIZE, input_features)
        offset_in = tl.arange(0, block_end - k)
        
        # Load input and weight blocks
        input_block = tl.load(input_ptr + index_val * input_features + offset_in, mask=offset_in < input_features, other=0.0)
        weight_block = tl.load(weight_ptr + offset_in * output_features, mask=offset_in < input_features, other=0.0)
        
        # Accumulate dot product
        acc += input_block * weight_block
    
    # Store result for this row
    tl.store(output_ptr + row_start + row_offset, acc, mask=mask)

@torch.fx.wrap
def optimized_linear_view_index(linear_out, index_tensor, weight_tensor):
    """
    Optimized version of linear + view + indexing operations
    """
    # Get tensor shapes
    input_features = weight_tensor.shape[1]
    output_features = weight_tensor.shape[0]
    batch_size = index_tensor.numel()
    
    # Create output tensor
    output_shape = (batch_size, output_features)
    output = torch.empty(output_shape, dtype=weight_tensor.dtype, device=weight_tensor.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 256  # Optimal for matrix multiplication
    num_programs = batch_size
    
    linear_view_index_kernel[(num_programs,)](
        input_ptr=linear_out,
        weight_ptr=weight_tensor,
        index_ptr=index_tensor,
        output_ptr=output,
        input_features=input_features,
        output_features=output_features,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_linear_view_index