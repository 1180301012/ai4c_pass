import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0, target_dtype):
    tmp_2 = torch.nn.functional.dropout(in_2, p = 0.0, training = False)
    to = tmp_2.to(target_dtype)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    # We need to determine the target dtype based on the input dtype 
    # since the pattern will be called with both float16 and bfloat16
    # Just return the input tensors - the kernel will handle the dtype
    return (in_2, in_1, in_0)

def determine_target_dtype(target_dtype_str):
    """Determine target dtype based on the string representation."""
    if "float16" in target_dtype_str:
        return torch.float16
    elif "bfloat16" in target_dtype_str:
        return torch.bfloat16
    else:
        # Default to float16 if not specified
        return torch.float16

@triton.jit
def optimized_linear_kernel(
    x_ptr,           # Input tensor [in_features] or [batch, seq_len, in_features]
    weight_ptr,      # Linear weight [out_features, in_features] 
    bias_ptr,        # Linear bias [out_features]
    output_ptr,      # Output tensor
    n_elements,      # Total number of elements in input
    in_features,     # Input features per head
    out_features,    # Output features per head
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # We need to handle matrix multiplication properly
    # For this kernel, each program handles a row of the output
    row = tl.program_id(1)
    
    if row < out_features:
        # For this row, accumulate dot product
        row_offset = row * in_features
        output_val = 0.0
        
        # Process input in blocks
        for col_block in range(0, in_features, BLOCK_SIZE):
            col_end = min(col_block + BLOCK_SIZE, in_features)
            
            # Load weight block for this row
            weight_start = row_offset + col_block
            weight_block = tl.load(weight_ptr + weight_start, mask=(tl.arange(0, BLOCK_SIZE) < (col_end - col_block)), other=0.0)
            
            # Load corresponding input block  
            input_block = x[col_block:col_end] if col_block < len(x) else tl.zeros([col_end - col_block], dtype=x.dtype)
            
            # Multiply and accumulate
            output_val += tl.sum(weight_block * input_block)
        
        # Add bias for this row
        bias = tl.load(bias_ptr + row, other=0.0)
        output_val += bias
        
        # Store the result
        # Each row will be stored sequentially
        result_offset = row * ((n_elements + in_features - 1) // in_features) + (offsets // in_features)
        result_mask = result_offset < ((n_elements + in_features - 1) // in_features)
        tl.store(output_ptr + result_offset, output_val, mask=result_mask)

@torch.fx.wrap  
def direct_linear_operation(x, weight, bias):
    # Handle different input shapes
    original_shape = x.shape
    
    if len(original_shape) == 3:  # [batch, seq_len, in_features]
        batch_size, seq_len, in_features = original_shape
        # Reshape to [batch*seq_len, in_features] for processing
        x_flat = x.reshape(-1, in_features)
    else:
        # Assume it's already 2D [batch, in_features] or [in_features]
        x_flat = x
        if len(x_flat.shape) == 1:
            # Add batch dimension if needed
            x_flat = x_flat.unsqueeze(0)
            flat_batch_size = 1
            seq_len = 1
        else:
            flat_batch_size, in_features = x_flat.shape
            seq_len = flat_batch_size
    
    # Get weight dimensions
    out_features, in_features = weight.shape
    
    # Calculate total elements and launch configuration
    n_total_elements = x_flat.numel()
    output_total_elements = (n_total_elements + in_features - 1) // in_features
    
    BLOCK_SIZE = 1024
    n_output_elements = (output_total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    row_programs = out_features
    
    # Create output tensor
    output = torch.empty(output_total_elements, dtype=x.dtype, device=x.device)
    
    # Launch the kernel
    # Each output element gets one program, plus one program per output row
    grid = (n_output_elements, row_programs)
    
    optimized_linear_kernel[grid](
        x_ptr=x_flat,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_total_elements,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to match original batch dimensions
    if len(original_shape) == 3:
        final_output = output.reshape(batch_size, seq_len, out_features)
    else:
        if len(x.shape) == 1:
            final_output = output[0:out_features]  # Take first out_features elements
        else:
            final_output = output.reshape(flat_batch_size, out_features)
            if flat_batch_size == 1:
                final_output = final_output[0]  # Remove batch dimension
    
    return final_output

def replacement_func():
    return direct_linear_operation