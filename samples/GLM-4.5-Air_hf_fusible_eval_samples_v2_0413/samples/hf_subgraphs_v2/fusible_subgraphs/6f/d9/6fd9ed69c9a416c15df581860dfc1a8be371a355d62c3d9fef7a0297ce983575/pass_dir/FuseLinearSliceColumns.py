import torch
import triton
import triton.language as tl

@triton.jit
def linear_slice_first_columns_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    n_batch,
    n_in_features,
    n_out_features,
    block_size: tl.constexpr,
):
    # Compute the first n_in_features columns of linear transformation
    batch_idx = tl.program_id(0)
    col_offset = tl.arange(0, n_in_features)
    
    # Load bias for first n_in_features columns
    bias = tl.load(bias_ptr + col_offset)
    
    # Load input row
    x_row = tl.load(x_ptr + batch_idx * n_in_features + tl.arange(0, n_in_features))
    
    # Load weight block (first n_in_features rows and columns)
    # Weight matrix is [n_out_features, n_in_features]
    weight_block = tl.load(weight_ptr + 
                          tl.arange(0, n_in_features)[:, None] * n_in_features + col_offset[None, :])
    
    # Compute result for first columns: W[:n_in_features, :] @ x_row + b[:n_in_features]
    result = tl.sum(x_row[None, :] * weight_block, dim=1) + bias
    
    # Store result
    tl.store(out_ptr + batch_idx * n_in_features + tl.arange(0, n_in_features), result)

@triton.jit
def linear_slice_last_columns_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr, 
    out_ptr,
    n_batch,
    n_in_features,
    n_out_features,
    block_size: tl.constexpr,
):
    # Compute the last n_in_features columns of linear transformation
    batch_idx = tl.program_id(0)
    
    # Calculate starting row for the last block
    start_row = n_out_features - n_in_features
    row_offset = tl.arange(start_row, n_out_features)
    
    # Load bias for last n_in_features columns
    bias = tl.load(bias_ptr + row_offset)
    
    # Load input row
    x_row = tl.load(x_ptr + batch_idx * n_in_features + tl.arange(0, n_in_features))
    
    # Load weight block (last n_in_features rows, all columns)
    # Weight matrix is [n_out_features, n_in_features]
    weight_block = tl.load(weight_ptr + 
                          row_offset[:, None] * n_in_features + tl.arange(0, n_in_features)[None, :])
    
    # Compute result for last columns: W[n_out_features-n_in_features:, :] @ x_row + b[n_out_features-n_in_features:]
    result = tl.sum(x_row[None, :] * weight_block, dim=1) + bias
    
    # Store result
    tl.store(out_ptr + batch_idx * n_in_features + tl.arange(0, n_in_features), result)

@torch.fx.wrap
def fused_linear_slice_first_columns(x, weight_1, bias_0):
    n_batch, n_in_features = x.shape
    n_out_features = weight_1.shape[0]
    
    out = torch.empty((n_batch, n_in_features), dtype=x.dtype, device=x.device)
    
    block_size = 256
    grid = (n_batch,)
    
    linear_slice_first_columns_kernel[grid](
        x,
        weight_1,
        bias_0, 
        out,
        n_batch,
        n_in_features,
        n_out_features,
        block_size
    )
    
    return out.view(-1, n_in_features)

@torch.fx.wrap
def fused_linear_slice_last_columns(x, weight_1, bias_0):
    n_batch, n_in_features = x.shape  
    n_out_features = weight_1.shape[0]
    
    out = torch.empty((n_batch, n_in_features), dtype=x.dtype, device=x.device)
    
    block_size = 256
    grid = (n_batch,)
    
    linear_slice_last_columns_kernel[grid](
        x,
        weight_1,
        bias_0,
        out,
        n_batch,
        n_in_features,
        n_out_features,
        block_size
    )
    
    return out.view(-1, n_in_features)

def pattern(input_tensor, weight_1, bias_0):
    linear = torch.nn.functional.linear(input_tensor, weight_1, bias_0)
    first_slice = linear[slice(None, None, None), slice(None, 256, None)]
    first_reshaped = first_slice.view(-1, 256)
    last_slice = linear[slice(None, None, None), slice(-256, None, None)]
    last_reshaped = last_slice.view(-1, 256)
    return first_reshaped, last_reshaped

def replacement_args(input_tensor, weight_1, bias_0):
    return (input_tensor, weight_1, bias_0, "first_last_columns")

@torch.fx.wrap
def dispatch_wrapper(input_tensor, weight_1, bias_0, route):
    if route == "first_last_columns":
        first_result = fused_linear_slice_first_columns(input_tensor, weight_1, bias_0)
        last_result = fused_linear_slice_last_columns(input_tensor, weight_1, bias_0)
        return first_result, last_result
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return lambda input_tensor, weight_1, bias_0, route: dispatch_wrapper(input_tensor, weight_1, bias_0, route)