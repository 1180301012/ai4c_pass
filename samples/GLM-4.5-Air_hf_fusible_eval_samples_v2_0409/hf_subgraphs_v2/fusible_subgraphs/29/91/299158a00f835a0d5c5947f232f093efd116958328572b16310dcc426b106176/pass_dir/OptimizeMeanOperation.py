import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern matching for mean operation: input_tensor.mean(-2)"""
    result = input_tensor.mean(-2)
    return result

def replacement_args(input_tensor):
    """Extract arguments for the mean operation replacement"""
    return (input_tensor,)

@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    n_batch, n_seq_len, n_features
):
    """Simple mean kernel using Triton for mean along dimension -2"""
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and feature for this program
    batch_idx = pid // n_features
    feature_idx = pid % n_features
    
    # Boundary check
    if batch_idx >= n_batch or feature_idx >= n_features:
        return
    
    # Sum across sequence dimension
    total = 0.0
    count = 0.0
    
    # Load elements for this batch and feature across sequence dimension
    for seq_idx in range(n_seq_len):
        offset = batch_idx * n_seq_len * n_features + seq_idx * n_features + feature_idx
        if offset < n_batch * n_seq_len * n_features:
            value = tl.load(input_ptr + offset, other=0.0)
            total += value
            count += 1.0
    
    # Calculate mean and store
    if count > 0:
        mean_val = total / n_seq_len
        output_offset = batch_idx * n_features + feature_idx
        tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def triton_mean(input_tensor):
    """Wrapper function to launch the optimized mean kernel"""
    # Get input tensor dimensions
    if input_tensor.dim() == 3:
        n_batch, n_seq_len, n_features = input_tensor.shape
    else:
        raise ValueError("Input tensor must be 3D for mean(-2) operation")
    
    # Create output tensor
    output = torch.empty((n_batch, n_features), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size - one program per output element
    grid_size = (n_batch * n_features + 1023) // 1024  # Round up to nearest multiple of 1024
    
    # Launch kernel
    mean_kernel[(grid_size,)](
        input_tensor, output,
        n_batch, n_seq_len, n_features
    )
    
    return output

def replacement_func():
    """Return the optimized mean function"""
    return triton_mean