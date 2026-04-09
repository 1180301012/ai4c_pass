import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.dropout(in_2, p = 0.0, training = False)
    to = tmp_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def optimized_linear_kernel_bf16(
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
    
    # Matrix multiplication: each program handles one output row
    row = tl.program_id(1)
    
    if row < out_features:
        # For this row, compute dot product
        row_offset = row * in_features
        result = 0.0
        
        # Process weight and input in blocks for better memory access
        for k in range(0, in_features, BLOCK_SIZE):
            k_end = min(k + BLOCK_SIZE, in_features)
            
            # Load weights for this row and block
            weights = tl.load(weight_ptr + row_offset + k, 
                            mask=(tl.arange(k, k_end) < in_features), 
                            other=0.0)
            
            # Load corresponding inputs
            inputs = tl.load(x_ptr + k, 
                           mask=(tl.arange(k, k_end) < in_features & offsets), 
                           other=0.0)
            
            # Multiply and accumulate
            result += tl.sum(weights * inputs)
        
        # Add bias
        bias = tl.load(bias_ptr + row, other=0.0)
        result += bias
        
        # Store result
        output_pos = row * ((n_elements + in_features - 1) // in_features) + (offsets // in_features)
        output_mask = output_pos < ((n_elements + in_features - 1) // in_features)
        tl.store(output_ptr + output_pos, result, mask=output_mask)

@torch.fx.wrap
def direct_linear_operation_bf16(x, weight, bias):
    # Get input shape and flatten if needed
    if len(x.shape) == 3:
        batch_size, seq_len, in_features = x.shape
        x_flat = x.reshape(-1, in_features)
    else:
        x_flat = x
        if len(x_flat.shape) == 1:
            x_flat = x_flat.unsqueeze(0)
    
    # Get dimensions
    out_features, in_features = weight.shape
    n_total_elements = x_flat.numel()
    
    # Launch configuration
    BLOCK_SIZE = 1024
    output_elements = (n_total_elements + in_features - 1) // in_features
    n_blocks = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (n_blocks, out_features)
    
    # Prepare output
    output = torch.empty(output_elements, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    optimized_linear_kernel_bf16[grid](
        x_ptr=x_flat,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_total_elements,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output
    if len(x.shape) == 3:
        return output.reshape(batch_size, seq_len, out_features)
    elif len(x.shape) == 1:
        return output[0:out_features]
    else:
        return output.reshape(x.shape[0], out_features)

def replacement_func():
    return direct_linear_operation_bf16