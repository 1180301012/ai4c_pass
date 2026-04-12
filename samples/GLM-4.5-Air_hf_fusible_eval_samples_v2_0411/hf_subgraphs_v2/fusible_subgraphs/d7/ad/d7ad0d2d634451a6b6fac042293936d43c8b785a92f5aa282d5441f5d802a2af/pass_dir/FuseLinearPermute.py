import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """ Match linear + permute pattern """
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_linear_permute_kernel(
    bias_ptr,
    weight_ptr, 
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    output_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ More efficient fused linear + permute kernel using Triton """
    # Each thread block processes a block of output elements
    pid = tl.program_id(0)
    
    # Calculate block start positions and offsets
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # Early return if block starts beyond total elements
    start_idx = pid * BLOCK_SIZE
    total_elements = output_dim * seq_len * batch_size
    if start_idx >= total_elements:
        return
    
    # Convert flat offset to batch/seq/output indices
    flat_idx = offsets
    batch_idx = flat_idx // (output_dim * seq_len)
    seq_idx = (flat_idx % (output_dim * seq_len)) // output_dim  
    out_idx = flat_idx % output_dim
    
    # Mask for valid elements in this block
    mask = offsets < total_elements
    
    # Process multiple elements in parallel
    # Use explicit dtype that matches the expected output
    results = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Vectorized bias loading
    bias_vals = tl.load(bias_ptr + out_idx, mask=out_idx < output_dim, other=0.0)
    results += bias_vals
    
    # Simple matrix multiplication loop
    for k in range(hidden_dim):
        # Load weight and input elements
        weight_val = tl.load(weight_ptr + out_idx * hidden_dim + k, 
                           mask=(out_idx < output_dim) & (k < hidden_dim), other=0.0)
        input_val = tl.load(input_ptr + k * (seq_len * batch_size) + 
                          seq_idx * batch_size + batch_idx,
                          mask=(k < hidden_dim) & (seq_idx < seq_len) & (batch_idx < batch_size), other=0.0)
        
        results += weight_val * input_val
    
    # Store results in permuted layout
    output_offsets = out_idx * (seq_len * batch_size) + seq_idx * batch_size + batch_idx
    tl.store(output_ptr + output_offsets, results, mask=mask)

@torch.fx.wrap
def fused_linear_permute(bias, weight, input_val):
    """ Fused linear + permute function """
    batch_size, seq_len, hidden_dim = input_val.shape
    output_dim = bias.shape[0]
    
    output_shape = (output_dim, seq_len, batch_size)
    output = torch.empty(output_shape, dtype=input_val.dtype, device=input_val.device)
    
    # Use larger block size for better GPU utilization
    BLOCK_SIZE = 256  # Process 256 elements per block
    
    # Calculate total number of elements and grid size
    total_elements = output_dim * seq_len * batch_size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel 
    fused_linear_permute_kernel[(grid_size,)](
        bias,
        weight, 
        input_val,
        output,
        batch_size,
        seq_len,
        hidden_dim,
        output_dim,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_linear_permute