import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Optimize torch.nn.functional.linear operation"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for linear operation optimization"""
    return (in_0, in_1, in_2)

@triton.jit
def linear_kernel(
    x_ptr,      # input tensor [batch, in_features]
    w_ptr,      # weight tensor [out_features, in_features] 
    b_ptr,      # bias tensor [out_features]
    out_ptr,    # output tensor [batch, out_features]
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Calculate program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    m_mask = m_offsets < batch_size
    n_mask = n_offsets < out_features
    k_mask = k_offsets < in_features
    
    # Load bias for this output column
    bias = tl.load(b_ptr + n_offsets[None, :], mask=n_mask[None, :], other=0.0)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension for matrix multiplication
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_ptr = x_ptr + m_offsets[:, None] * in_features + k
        w_ptr_k = w_ptr + n_offsets[None, :] * in_features + k
        
        # Load input and weight blocks
        x_block = tl.load(k_ptr, mask=m_mask[:, None] & (k_offsets[None, :] < in_features), other=0.0)
        w_block = tl.load(w_ptr_k, mask=n_mask[:, None] & (k_offsets[None, :] < in_features), other=0.0)
        
        # Matrix multiply step
        accumulator += x_block[:, :, None] * w_block[None, :, :]
    
    # Sum over k dimension and add bias
    accumulator = tl.sum(accumulator, axis=1) + bias[None, :]
    
    # Store result
    out_ptr_base = out_ptr + m_offsets[:, None] * out_features + n_offsets[None, :]
    tl.store(out_ptr_base, accumulator, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def triton_linear(x, w, b):
    """High-performance linear layer using Triton"""
    # Handle different input tensor shapes
    if x.dim() == 2:
        batch_size, in_features = x.shape
    elif x.dim() == 1:
        # Handle case where input might be 1D
        batch_size = 1
        in_features = x.shape[0]
        x = x.unsqueeze(0)  # Add batch dimension
    else:
        # Fallback to original PyTorch implementation for unexpected shapes
        return torch.nn.functional.linear(x, w, b)
    
    out_features = w.shape[0]
    
    # Set block sizes based on typical GPU architecture
    BLOCK_SIZE_M = 64   # Batch block size
    BLOCK_SIZE_N = 32   # Output features block size  
    BLOCK_SIZE_K = 32   # Input features block size
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Allocate output
    out = torch.empty((batch_size, out_features), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    linear_kernel[(grid_m, grid_n)](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    """Returns the optimized linear function"""
    return triton_linear