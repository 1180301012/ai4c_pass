import torch
import triton
import triton.language as tl

def pattern(x, y, z, w):
    # Pattern for: addition + dropout + layer_norm
    # x: in_0 (bias), y: in_1 (weight), z: in_2, w: in_3
    added = z + w
    dropped = torch.nn.functional.dropout(added, 0.1, False, False)
    normalized = torch.nn.functional.layer_norm(dropped, (None,), y, x, 1e-12)
    return (dropped, normalized)

def replacement_args(x, y, z, w):
    return (x, y, z, w)

@triton.jit
def fused_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    norm_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data - stride across hidden_size dimension
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    weight_data = tl.load(weight_ptr + offsets % hidden_size, mask=(offsets % hidden_size) < hidden_size)
    bias_data = tl.load(bias_ptr + offsets % hidden_size, mask=(offsets % hidden_size) < hidden_size)
    
    # Combined computation: input * norm_scale (from dropout) * weight + bias
    # Dropout with p=0.1, training=False = multiply by 0.9
    # Layer norm = (input - mean) / std * weight + bias
    # Combined: (input * 0.9 - mean) / std * weight + bias
    
    # Compute mean and variance for batch normalization
    mean = tl.sum(input_data * norm_scale) / tl.sum(mask)
    x_centered = input_data * norm_scale - mean
    var = tl.sum(x_centered * x_centered) / tl.sum(mask)
    std = tl.sqrt(var + 1e-12)
    
    # Normalized and scaled computation
    normalized = x_centered / std
    output = normalized * weight_data + bias_data
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def fused_layer_norm_with_dropout_elimination(input_tensor, weight_tensor, bias_tensor):
    # Get input dimensions
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1
    hidden_size = input_tensor.shape[-1]
    
    # Calculate total elements
    n_elements = input_tensor.numel()
    
    # Choose appropriate block size for different hidden sizes
    if hidden_size >= 1024:
        BLOCK_SIZE = 1024
    elif hidden_size >= 512:
        BLOCK_SIZE = 512
    elif hidden_size >= 256:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 128
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel with dropout scaling factor incorporated
    fused_layer_norm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        norm_scale=0.9,  # Dropout with p=0.1, training=False = * 0.9
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # The dropout output is the scaled input (input * 0.9)
    dropout_output = input_tensor * 0.9
    
    # For the original dropout output (tmp_3) and layer_norm output (tmp_4)
    return (dropout_output, output)

def replacement_func():
    return fused_layer_norm_with_dropout_elimination