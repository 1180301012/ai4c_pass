import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Pattern matching for linear operation: torch.nn.functional.linear(input, weight, bias)"""
    result = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    return result

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    """Extract arguments for the linear operation replacement"""
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def linear_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    """Simplified linear kernel using Triton for matrix multiplication + bias"""
    # Get program ID  
    pid = tl.program_id(0)
    
    # Calculate row and column for this program
    row_idx = pid // N
    col_idx = pid % N
    
    # Boundary check
    if row_idx >= M or col_idx >= N:
        return
    
    # Initialize accumulator with bias
    offset = col_idx
    mask = col_idx < N
    acc = tl.load(bias_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    
    # Matrix multiplication vector dot product
    for k in range(K):
        # Load elements from input and weight tensors
        x_offset = row_idx * K + k
        w_offset = col_idx * K + k
        x_mask = x_offset < (M * K)
        w_mask = w_offset < (N * K)
        
        x_val = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
        w_val = tl.load(weight_ptr + w_offset, mask=w_mask, other=0.0)
        
        # Multiply and accumulate
        acc += x_val * w_val
    
    # Store result
    out_offset = row_idx * N + col_idx
    out_mask = out_offset < (M * N)
    tl.store(out_ptr + out_offset, acc, mask=out_mask)

@torch.fx.wrap  
def triton_linear(input_tensor, weight_tensor, bias_tensor):
    """Wrapper function to launch the optimized linear kernel"""
    n_rows, n_features = input_tensor.shape
    n_cols = weight_tensor.shape[0]
    
    # Create output tensor
    output = torch.empty((n_rows, n_cols), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set grid size - one program per output element  
    grid_size = (n_rows * n_cols + 1023) // 1024  # Round up to nearest multiple of 1024
    
    # Launch kernel
    linear_kernel[(grid_size,)](
        input_tensor, weight_tensor, bias_tensor, output,
        n_rows, n_cols, n_features
    )
    
    return output

def replacement_func():
    """Return the optimized linear function"""
    return triton_linear