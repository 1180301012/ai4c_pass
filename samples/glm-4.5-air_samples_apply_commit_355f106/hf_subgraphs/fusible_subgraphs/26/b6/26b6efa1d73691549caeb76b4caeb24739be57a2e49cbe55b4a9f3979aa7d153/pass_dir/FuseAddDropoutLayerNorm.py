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
def fully_fused_kernel(
    input_x_ptr,    # in_2
    input_z_ptr,    # in_3  
    weight_ptr,     # in_1 (layer_norm weight)
    bias_ptr,       # in_0 (layer_norm bias)
    dropout_out_ptr, # output for dropout result
    layer_norm_out_ptr, # output for layer_norm result
    n_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load addition inputs: in_2 + in_3
    x_data = tl.load(input_x_ptr + offsets, mask=mask, other=0.0)
    z_data = tl.load(input_z_ptr + offsets, mask=mask, other=0.0)
    added_data = x_data + z_data
    
    # Dropout with p=0.1, training=False = scaling by 0.9
    dropout_output = added_data * 0.9
    
    # Load layer norm parameters (broadcast across batch and sequence)
    weight_data = tl.load(weight_ptr + offsets % hidden_size, mask=(offsets % hidden_size) < hidden_size)
    bias_data = tl.load(bias_ptr + offsets % hidden_size, mask=(offsets % hidden_size) < hidden_size)
    
    # Layer normalization with dropout scaling incorporated
    # Compute mean and variance over the normalized shape dimension
    mean = tl.sum(added_data * 0.9) / tl.sum(mask)
    x_centered = added_data * 0.9 - mean
    var = tl.sum(x_centered * x_centered) / tl.sum(mask)
    std = tl.sqrt(var + 1e-12)
    
    # Apply layer normalization
    normalized = x_centered / std
    layer_norm_output = normalized * weight_data + bias_data
    
    # Store both results
    tl.store(dropout_out_ptr + offsets, dropout_output, mask=mask)
    tl.store(layer_norm_out_ptr + offsets, layer_norm_output, mask=mask)

@torch.fx.wrap
def fully_fused_computation(input_x, input_z, weight, bias):
    # Get input dimensions
    batch_size = input_x.shape[0]
    seq_len = input_x.shape[1] if len(input_x.shape) > 1 else 1
    hidden_size = input_x.shape[-1]
    
    # Calculate total elements
    n_elements = input_x.numel()
    
    # Optimized block size configuration based on hidden_size
    if hidden_size >= 1024:
        BLOCK_SIZE = 2048  # Larger block for larger hidden sizes
    elif hidden_size >= 512:
        BLOCK_SIZE = 1024
    elif hidden_size >= 256:
        BLOCK_SIZE = 512
    elif hidden_size >= 128:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 128  # Smaller block for smaller hidden sizes
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    dropout_output = torch.empty_like(input_x)
    layer_norm_output = torch.empty_like(input_x)
    
    # Launch the fully fused kernel
    fully_fused_kernel[(num_programs,)](
        input_x_ptr=input_x,
        input_z_ptr=input_z,
        weight_ptr=weight,
        bias_ptr=bias,
        dropout_out_ptr=dropout_output,
        layer_norm_out_ptr=layer_norm_output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return both outputs (tmp_3 = dropout output, tmp_4 = layer_norm output)
    return (dropout_output, layer_norm_output)

def replacement_func():
    return fully_fused_computation