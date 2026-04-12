import torch
import triton
import triton.language as tl

# Pattern matching function - matches the layer normalization
def pattern(input_tensor, weight, bias):
    result = torch.nn.functional.layer_norm(input_tensor, (-1,), weight, bias, 1e-05)
    return result

# Argument extraction function
def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

# Optimized Triton kernel for layer normalization with weight and bias
@triton.jit
def optimized_layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_0,
    stride_1,
    stride_2,
    x_size,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Load input slice for current position
    x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance
    x_mean = tl.sum(x_data, mask=mask) / hidden_size
    x_var = tl.sum((x_data - x_mean) * (x_data - x_mean), mask=mask) / hidden_size
    
    # Apply normalization
    x_norm = (x_data - x_mean) / tl.sqrt(x_var + eps)
    
    # Load weights and bias
    wt_data = tl.load(w_ptr + offsets, mask=mask, other=1.0)
    bias_data = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Apply affine transformation
    out_data = x_norm * wt_data + bias_data
    
    # Store result
    tl.store(out_ptr + offsets, out_data, mask=mask)

@triton.jit
def optimized_layernorm_kernel_parallel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_0,
    stride_1,
    stride_2,
    batch_size,
    seq_len,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one hidden dimension position
    hid_pid = tl.program_id(0)
    block_start = hid_pid * BLOCK_SIZE
    hid_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    hid_mask = hid_offsets < hidden_size
    
    batch_pid = tl.program_id(1)
    seq_pid = tl.program_id(2)
    
    # Calculate base address for current batch and sequence position
    base_offset = batch_pid * stride_0 + seq_pid * stride_1
    
    # Load input slice for current position
    x_data = tl.load(x_ptr + base_offset + hid_offsets * stride_2, mask=hid_mask, other=0.0)
    
    # Compute mean and variance
    x_mean = tl.sum(x_data, mask=hid_mask) / hidden_size
    x_var = tl.sum((x_data - x_mean) * (x_data - x_mean), mask=hid_mask) / hidden_size
    
    # Apply normalization
    x_norm = (x_data - x_mean) / tl.sqrt(x_var + eps)
    
    # Load weights and bias (same for all positions)
    wt_data = tl.load(w_ptr + hid_offsets, mask=hid_mask, other=1.0)
    bias_data = tl.load(b_ptr + hid_offsets, mask=hid_mask, other=0.0)
    
    # Apply affine transformation
    out_data = x_norm * wt_data + bias_data
    
    # Store result
    tl.store(out_ptr + base_offset + hid_offsets * stride_2, out_data, mask=hid_mask)

# Kernel wrapper for optimized layer normalization
@torch.fx.wrap
def optimized_layer_norm(input_tensor, weight, bias):
    input_shape = input_tensor.shape
    batch_size, seq_len, hidden_size = input_shape
    
    BLOCK_SIZE = 1024
    num_blocks = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    # Use parallel kernel for better performance
    optimized_layernorm_kernel_parallel[
        (num_blocks, batch_size, seq_len)
    ](
        x_ptr=input_tensor,
        w_ptr=weight,
        b_ptr=bias,
        out_ptr=output,
        stride_0=input_shape[1] * input_shape[2],
        stride_1=input_shape[2],
        stride_2=1,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_layer_norm