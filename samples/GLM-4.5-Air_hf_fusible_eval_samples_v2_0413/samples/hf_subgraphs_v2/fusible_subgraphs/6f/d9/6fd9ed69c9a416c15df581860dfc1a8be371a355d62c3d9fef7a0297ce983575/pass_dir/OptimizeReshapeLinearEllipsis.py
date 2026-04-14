import torch
import triton
import triton.language as tl

@triton.jit
def reshape_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    first_out_ptr,
    last_out_ptr, 
    n_batch,
    n_seq,
    n_in_features,
    n_out_features,
    block_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate input offsets
    input_offset = (batch_idx * n_seq + seq_idx) * n_in_features
    weight_offset = 0
    
    # Load input vector for this batch and sequence position
    x_vec = tl.load(input_ptr + input_offset + tl.arange(0, n_in_features))
    
    # Load weights for first output column
    first_weight = tl.load(weight_ptr + tl.arange(0, n_in_features))
    
    # Load bias for first output column
    first_bias = tl.load(bias_ptr + 0)
    
    # Compute first output: W[:, 0] @ x + b[0]
    first_out = tl.sum(x_vec * first_weight) + first_bias
    
    # Load weights for last output column
    last_weight = tl.load(weight_ptr + (n_out_features - 1) * n_in_features + tl.arange(0, n_in_features))
    
    # Load bias for last output column  
    last_bias = tl.load(bias_ptr + n_out_features - 1)
    
    # Compute last output: W[:, -1] @ x + b[-1]
    last_out = tl.sum(x_vec * last_weight) + last_bias
    
    # Store results
    first_output_offset = (batch_idx * n_seq + seq_idx) * 1  # single column
    last_output_offset = (batch_idx * n_seq + seq_idx) * 1   # single column
    
    tl.store(first_out_ptr + first_output_offset, first_out)
    tl.store(last_out_ptr + last_output_offset, last_out)

@torch.fx.wrap
def optimized_reshape_linear_ellipsis(reshape_input, weight_3, bias_2):
    # Reshape input to [300, 1, 256]
    reshaped = reshape_input.view(300, -1, 256)
    
    n_batch, n_seq, n_in_features = reshaped.shape
    n_out_features = weight_3.shape[0]
    
    # Create output tensors
    first_out = torch.empty((n_batch, n_seq), dtype=reshaped.dtype, device=reshaped.device)
    last_out = torch.empty((n_batch, n_seq), dtype=reshaped.dtype, device=reshaped.device)
    
    # Launch kernel
    grid = (n_batch, n_seq)
    
    reshape_linear_kernel[grid](
        reshaped,
        weight_3,
        bias_2,
        first_out,
        last_out,
        n_batch,
        n_seq,
        n_in_features,
        n_out_features,
        256
    )
    
    # Add the dimension back to match original output shape [300, 1, 256]
    first_out_expanded = first_out.unsqueeze(-1)
    last_out_expanded = last_out.unsqueeze(-1)
    
    return first_out_expanded, last_out_expanded

def pattern(reshape_input, weight_3, bias_2):
    tmp_9 = reshape_input.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, weight_3, bias_2)
    tmp_11 = linear_1[Ellipsis, slice(None, 256, None)]
    tmp_12 = linear_1[Ellipsis, slice(-256, None, None)]
    return tmp_11, tmp_12

def replacement_args(reshape_input, weight_3, bias_2):
    return (reshape_input, weight_3, bias_2, "reshape_linear_ellipsis")

@torch.fx.wrap
def dispatch_wrapper(reshape_input, weight_3, bias_2, route):
    if route == "reshape_linear_ellipsis":
        return optimized_reshape_linear_ellipsis(reshape_input, weight_3, bias_2)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return lambda reshape_input, weight_3, bias_2, route: dispatch_wrapper(reshape_input, weight_3, bias_2, route)