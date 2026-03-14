import torch
import triton
import triton.language as tl

# Pattern matching function - matches just softmax + dropout before bmm
def pattern(in_0, in_1):
    # Match just the softmax and dropout operations
    tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
    # Return the dropout result as the output we want to replace
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for softmax when dropout p=0.0 (no-op)
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply softmax - along the last dimension
    # For 3D tensors [batch_size, seq_len, feature_dim], we process along feature_dim
    if offsets.shape[0] > 1:
        # Reshape for softmax along last dimension
        reshaped_x = x.reshape(-1, x.shape[-1])
        max_vals = tl.max(reshaped_x, axis=1)
        exp_x = tl.exp(reshaped_x - max_vals[:, None])
        sum_exp = tl.sum(exp_x, axis=1)
        softmax_result = exp_x / sum_exp[:, None]
        output = softmax_result.reshape(x.shape)
    else:
        output = x
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_softmax_dropout_forward(in_0, in_1):
    # Since dropout with p=0.0 is no-op, we just return softmax computed with Triton
    n_elements = in_0.numel()
    output = torch.empty_like(in_0)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    softmax_kernel[(num_programs,)](
        in_0,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_softmax_dropout_forward

# Pattern matching function - matches just softmax + dropout before bmm
def pattern(in_0, in_1):
    # Match just the softmax and dropout operations
    tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
    # Return the dropout result as the output we want to replace
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for softmax when dropout p=0.0 (no-op)
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply softmax - along the last dimension
    # For 3D tensors [batch_size, seq_len, feature_dim], we process along feature_dim
    if offsets.shape[0] > 1:
        # Reshape for softmax along last dimension
        reshaped_x = x.reshape(-1, x.shape[-1])
        max_vals = tl.max(reshaped_x, axis=1)
        exp_x = tl.exp(reshaped_x - max_vals[:, None])
        sum_exp = tl.sum(exp_x, axis=1)
        softmax_result = exp_x / sum_exp[:, None]
        output = softmax_result.reshape(x.shape)
    else:
        output = x
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def optimized_softmax_dropout_forward(in_0, in_1):
    # Since dropout with p=0.0 is no-op, we just return softmax computed with Triton
    n_elements = in_0.numel()
    output = torch.empty_like(in_0)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    softmax_kernel[(num_programs,)](
        in_0,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_softmax_dropout_forward