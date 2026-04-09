import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def dropout_linear_kernel(
    x_ptr,           # Input tensor
    weight_ptr,      # Linear weight [out_features, in_features]  
    bias_ptr,        # Linear bias [out_features]
    output_ptr,      # Output tensor
    n_elements,      # Total number of elements in input
    in_features,     # Input features per head
    out_features,    # Output features per head
    dropout_p,       # Dropout probability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input element with proper dtype
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout (scaled by 1/(1-p) during training)
    if dropout_p > 0:
        # Create dropout mask using tensor operations that preserve dtype
        random_vals = (offsets * 12345) % 1000 > (dropout_p * 1000)
        keep_mask = random_vals.to(x.dtype)
        dropout_scale = tl.cast(1.0 / (1.0 - dropout_p), x.dtype)
        x = x * keep_mask * dropout_scale
    
    # Simplified matrix multiplication using 2D launch configuration
    row = tl.program_id(1)
    col = tl.program_id(0)
    
    if row < out_features and col < (n_elements + in_features - 1) // in_features:
        # Compute dot product for this output element
        output_val = 0.0
        
        # Get the flattened input index for this output element
        input_start = col * in_features
        
        # Compute dot product
        for k in range(in_features):
            weight_offset = row * in_features + k
            input_offset = input_start + k
            
            if input_offset < n_elements:
                weight_val = tl.load(weight_ptr + weight_offset)
                input_val = tl.load(x_ptr + input_offset)
                output_val += weight_val * input_val
        
        # Add bias
        bias_val = tl.load(bias_ptr + row)
        output_val += bias_val
        
        # Store result
        output_offset = row * ((n_elements + in_features - 1) // in_features) + col
        tl.store(output_ptr + output_offset, output_val)

@torch.fx.wrap  
def fused_dropout_linear(x, weight, bias):
    # Get tensor properties
    in_shape = x.shape
    out_features, in_features = weight.shape
    dtype = x.dtype
    
    # Reshape if needed to handle batch dimension properly
    if len(in_shape) == 3:  # [batch, seq_len, features]
        batch_size, seq_len, _ = in_shape
        x_flat = x.reshape(-1, in_features)  # [batch*seq_len, in_features]
    else:
        x_flat = x
        batch_size, seq_len = 1, x_flat.shape[0] // in_features
    
    n_total_elements = x_flat.numel()
    
    # Launch configuration: one program per output element
    output_elements = (n_total_elements + in_features - 1) // in_features
    grid = (output_elements, out_features)
    
    output = torch.empty(output_elements * out_features, dtype=dtype, device=x.device)
    
    dropout_linear_kernel[grid](
        x_ptr=x_flat,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_total_elements,
        in_features=in_features,
        out_features=out_features,
        dropout_p=0.1,
        BLOCK_SIZE=128,  # Reduced block size for better compatibility
    )
    
    # Reshape output to match expected dimensions
    if len(in_shape) == 3:
        # Reshape to [batch, seq_len, out_features]
        output_reshaped = output.reshape(batch_size, seq_len, out_features)
    else:
        # Reshape to [batch, out_features]
        output_reshaped = output.reshape(batch_size, out_features)
        if batch_size == 1:
            output_reshaped = output_reshaped[0]  # Remove batch dimension
    
    return output_reshaped

def replacement_func():
    return fused_dropout_linear