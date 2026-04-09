import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    # BigBird pattern: dropout(0.1) + linear
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def efficient_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features,
    out_features,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes multiple output features for one batch element
    batch_id = tl.program_id(0)
    
    if batch_id >= batch_size:
        return
    
    # Vectorized processing of multiple output features
    output_offset = batch_id * out_features + tl.arange(0, BLOCK_SIZE)
    mask = output_offset < (batch_id + 1) * out_features
    
    # Initialize output values for all features in this block
    output_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute dot product for each output feature in parallel
    for k in range(in_features):
        # Load current input element
        input_val = tl.load(x_ptr + batch_id * in_features + k)
        
        # Load weight block: [BLOCK_SIZE]
        weight_vals = tl.load(weight_ptr + output_offset % out_features * in_features + k,
                            mask=mask, other=0.0)
        
        # Accumulate dot product
        output_vals += input_val * weight_vals
    
    # Add bias (broadcast to all elements in block)
    bias_vals = tl.load(bias_ptr + output_offset % out_features, mask=mask, other=0.0)
    output_vals = output_vals + bias_vals
    
    # Apply dropout scaling compensation
    output_vals = output_vals * (1.0 / 0.9)
    
    # Store results
    tl.store(output_ptr + output_offset, output_vals, mask=mask)

@torch.fx.wrap
def fused_dropout_linear(x, weight, bias):
    # Get tensor properties
    in_shape = x.shape
    out_features, in_features = weight.shape
    
    # Flatten input to 2D: [batch*seq_len, in_features]
    if len(in_shape) == 1:
        x_flat = x.unsqueeze(0)  # [1, in_features]
        final_shape = out_features
    elif len(in_shape) == 2:
        x_flat = x  # [batch, in_features]
        final_shape = x.shape[0], out_features
    else:  # 3D [batch, seq_len, in_features]
        batch_size, seq_len = in_shape[0], in_shape[1]
        x_flat = x.reshape(-1, in_features)  # [batch*seq_len, in_features]
        final_shape = batch_size, seq_len, out_features
    
    n_batch = x_flat.shape[0]
    
    # Prepare output using only allowed tensor allocation
    output = torch.empty(n_batch * out_features, dtype=x.dtype, device=x.device)
    
    # Launch configuration - one program per batch element
    grid = (n_batch,)
    
    # Execute efficient kernel with block processing
    efficient_linear_kernel[grid](
        x_ptr=x_flat,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        in_features=in_features,
        out_features=out_features,
        batch_size=n_batch,
        BLOCK_SIZE=128,  # Process 128 output features in parallel
    )
    
    # Reshape output to match input dimensions
    if len(in_shape) == 1:
        return output[0:out_features]
    elif len(in_shape) == 2:
        return output.reshape(x.shape[0], out_features)
    else:  # 3D
        return output.reshape(batch_size, seq_len, out_features)

def replacement_func():
    return fused_dropout_linear