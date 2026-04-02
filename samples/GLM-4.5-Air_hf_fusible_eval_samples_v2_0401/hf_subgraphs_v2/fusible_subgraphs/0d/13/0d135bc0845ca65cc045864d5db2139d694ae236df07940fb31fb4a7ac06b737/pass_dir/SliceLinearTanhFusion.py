import torch
import triton
import triton.language as tl

def pattern(layer_norm_out, weight, bias):
    # Slice first sequence element
    sliced = layer_norm_out[(slice(None, None, None), 0)]
    # Linear transformation
    linear = sliced @ weight + bias
    # Tanh activation
    result = torch.tanh(linear)
    return result

def replacement_args(tmp_6, in_4, in_3):
    return (tmp_6, in_4, in_3)

@triton.jit
def slice_linear_tanh_kernel(
    layer_norm_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    hidden_dim,
    output_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch
    if tl.program_id(0) >= batch_size:
        return
    
    # We only need the first sequence element (index 0)
    seq_idx = 0
    
    # Compute offset for this batch element
    batch_offset = tl.program_id(0) * hidden_dim
    
    # Load weight and bias (these are constant across all elements)
    if tl.program_id(0) == 0:  # Only once per program
        # Load first row of weight matrix [hidden_dim, output_dim]
        for k in range(0, hidden_dim, BLOCK_SIZE):
            weight_ptrs = weight_ptr + k * output_dim
            weight_mask = k + tl.arange(0, BLOCK_SIZE) < hidden_dim
            weight_data = tl.load(weight_ptrs + tl.arange(0, BLOCK_SIZE)[:, None] * output_dim, 
                                mask=weight_mask[:, None], other=0.0)
            
            # Load corresponding bias values
            bias_ptrs = bias_ptr + tl.arange(0, BLOCK_SIZE)
            bias_mask = k + tl.arange(0, BLOCK_SIZE) < hidden_dim
            bias_data = tl.load(bias_ptrs + k, mask=bias_mask, other=0.0)
            
            # Process each output dimension
            for j in range(0, output_dim, BLOCK_SIZE):
                j_mask = j + tl.arange(0, BLOCK_SIZE) < output_dim
                j_range = j + tl.arange(0, BLOCK_SIZE)
                
                # Linear computation: input @ weight_row + bias
                # Here "input" is just 1.0 since we're taking the slice directly
                # This is equivalent to just taking the weight row
                weighted = weight_data * 1.0
                
                # Store the result for this output dimension
                out_ptrs = out_ptr + batch_offset + j_range
                tl.store(out_ptrs + tl.arange(0, BLOCK_SIZE)[:, None], 
                        weighted + bias_data[:, None], mask=j_mask[:, None])

@triton.jit
def simplified_slice_linear_kernel(
    layer_norm_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    batch_size,
    hidden_dim,
    seq_idx,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    # For the first sequence element, we take the weight matrix row directly
    # This is equivalent to: layer_norm_out[pid, seq_idx, :] @ weight + bias
    # But since we're slicing, it's just weight[seq_idx, :] + bias
    
    col_offset = pid * hidden_dim
    
    # Load weights and bias for this hidden dimension
    for j in range(0, hidden_dim, BLOCK_SIZE):
        j_mask = j + tl.arange(0, BLOCK_SIZE) < hidden_dim
        j_range = j + tl.arange(0, BLOCK_SIZE)
        
        # Load weight row (first sequence element)
        weight_data = tl.load(weight_ptr + j_range * hidden_dim, mask=j_mask, other=0.0)
        bias_data = tl.load(bias_ptr + j_range, mask=j_mask, other=0.0)
        
        # Apply tanh activation
        result = tl.tanh(weight_data + bias_data)
        
        # Store result
        out_ptrs = out_ptr + col_offset + j_range
        tl.store(out_ptrs + tl.arange(0, BLOCK_SIZE), result, mask=j_mask)

@triton.jit
def slice_linear_tanh_kernel(
    layer_norm_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    seq_idx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    # Load the first sequence element for this batch
    # layer_norm_out[pid, seq_idx, :]
    col_offset = pid * seq_len * hidden_dim + seq_idx * hidden_dim
    
    # Linear transformation: sum over hidden dimension, result is [batch_size, output_dim]
    # Since we're using [hidden_dim, hidden_dim] weight matrix, output_dim = hidden_dim
    output_offset = pid * hidden_dim
    
    for j in range(0, hidden_dim, BLOCK_SIZE):
        j_mask = j + tl.arange(0, BLOCK_SIZE) < hidden_dim
        j_range = j + tl.arange(0, BLOCK_SIZE)
        
        # Load weights for current output dimension: weight[:, j]
        # Weight shape is [hidden_dim, hidden_dim]
        weight_ptrs = weight_ptr + j_range * hidden_dim
        weight_data = tl.load(weight_ptrs + tl.arange(0, BLOCK_SIZE)[:, None], 
                            mask=j_mask[:, None], other=0.0)
        
        # Load corresponding bias values
        bias_ptrs = bias_ptr + j_range
        bias_data = tl.load(bias_ptrs, mask=j_mask, other=0.0)
        
        # Load layer norm values for first sequence element: layer_norm_out[pid, seq_idx, :]
        layer_norm_ptrs = layer_norm_ptr + col_offset + j_range
        layer_norm_data = tl.load(layer_norm_ptrs, mask=j_mask, other=0.0)
        
        # Linear computation: sum(layer_norm_data * weight_data) + bias_data
        # We need to compute dot product and add broadcast bias
        result = tl.sum(layer_norm_data * weight_data, axis=0) + bias_data
        
        # Apply tanh activation
        result = tl.tanh(result)
        
        # Store result
        out_ptrs = out_ptr + output_offset + j_range
        tl.store(out_ptrs, result, mask=j_mask)

@triton.jit
def slice_linear_kernel(
    layer_norm_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    seq_idx,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output dimension for all batch elements
    pid = tl.program_id(1)  # Outer loop over output dimensions
    j = pid
    if j >= hidden_dim:
        return
    
    # Each warp handles one batch element
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    # Base offset for this batch element
    input_base_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
    output_base_offset = batch_idx * hidden_dim
    
    # Load linear weight for output dimension j: weight[:, j]
    weight_data = tl.load(weight_ptr + tl.arange(0, hidden_dim) * hidden_dim + j, 
                        mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
    
    # Load bias for output dimension j
    bias_data = tl.load(bias_ptr + j, mask=j < hidden_dim, other=0.0)
    
    # Load layer norm values for first sequence element: layer_norm_out[batch_idx, seq_idx, :]
    layer_norm_data = tl.load(layer_norm_ptr + input_base_offset + tl.arange(0, hidden_dim),
                             mask=tl.arange(0, hidden_dim) < hidden_dim, other=0.0)
    
    # Linear computation: sum(layer_norm_data * weight_data) + bias_data
    result = tl.sum(layer_norm_data * weight_data) + bias_data
    
    # Apply tanh activation
    result = tl.tanh(result)
    
    # Store result for output dimension j
    output_offset = output_base_offset + j
    tl.store(out_ptr + output_offset, result)

@torch.fx.wrap  
def slice_linear_tanh(layer_norm_out, weight, bias):
    batch_size = layer_norm_out.shape[0]
    seq_len = layer_norm_out.shape[1]
    hidden_dim = layer_norm_out.shape[2]
    
    out = torch.empty((batch_size, hidden_dim), dtype=layer_norm_out.dtype, device=layer_norm_out.device)
    
    BLOCK_SIZE = 1024
    # Grid: (batch_size, hidden_dim) - one program per batch element per output dimension
    grid = (batch_size, hidden_dim)
    
    # We only need the first sequence element (index 0)
    seq_idx = 0
    
    slice_linear_kernel[grid](
        layer_norm_ptr=layer_norm_out,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        seq_idx=seq_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return slice_linear_tanh