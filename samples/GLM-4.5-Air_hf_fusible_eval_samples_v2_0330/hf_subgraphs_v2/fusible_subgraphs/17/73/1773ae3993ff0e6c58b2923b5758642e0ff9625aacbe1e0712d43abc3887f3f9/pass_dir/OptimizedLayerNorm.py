import torch
import triton
import triton.language as tl

# Pattern matching function - matches the layer normalization operation
def pattern(in_4, in_1, in_0):
    """
    Match the layer normalization operation from the computation graph
    """
    tmp_3 = torch.nn.functional.layer_norm(in_4, in_4.shape[-1:], in_1, in_0, 1e-12)
    return tmp_3

# Argument extraction function
def replacement_args(in_4, in_1, in_0):
    return (in_4, in_1, in_0)

# Optimized Triton kernel for layer normalization
@triton.jit
def layernorm_kernel(
    x_ptr,         # Pointer to input tensor
    gamma_ptr,     # Pointer to scale (weight) tensor  
    beta_ptr,      # Pointer to shift (bias) tensor
    out_ptr,       # Pointer to output tensor
    n_elements,    # Total number of elements
    hidden_size,   # Size of hidden dimension
    BLOCK_M: tl.constexpr,      # Block size for M dimension (batch * seq_len)
    BLOCK_N: tl.constexpr       # Block size for N dimension (hidden_size)
):
    # Program ID determines which block we process
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Calculate memory offsets
    x_base = m * hidden_size + n * BLOCK_N
    x_offsets = x_base + tl.arange(0, BLOCK_N)
    
    # Load input data with bounds checking
    x = tl.load(x_ptr + x_offsets, mask=x_offsets < n_elements, other=0.0)
    
    # Load scale and shift parameters
    gamma = tl.load(gamma_ptr + n * BLOCK_N + tl.arange(0, BLOCK_N), 
                   mask=n * BLOCK_N + tl.arange(0, BLOCK_N) < hidden_size, other=1.0)
    beta = tl.load(beta_ptr + n * BLOCK_N + tl.arange(0, BLOCK_N), 
                  mask=n * BLOCK_N + tl.arange(0, BLOCK_N) < hidden_size, other=0.0)
    
    # Compute mean (broadcast to all elements in the block)
    mean = tl.sum(x) / hidden_size
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered) / hidden_size
    
    # Normalize with epsilon
    std_inv = 1.0 / tl.sqrt(variance + 1e-12)
    x_normalized = x_centered * std_inv
    
    # Apply scale and shift
    out = x_normalized * gamma + beta
    
    # Store result
    tl.store(out_ptr + x_offsets, out, mask=x_offsets < n_elements)

@torch.fx.wrap
def optimized_layernorm(x, gamma, beta):
    """
    Optimized layer normalization using Triton
    """
    n_elements = x.numel()
    hidden_size = x.shape[-1]
    
    # Choose optimal block sizes based on hidden_size
    if hidden_size <= 64:
        BLOCK_N = 64
    elif hidden_size <= 128:
        BLOCK_N = 128
    elif hidden_size <= 256:
        BLOCK_N = 256
    elif hidden_size <= 512:
        BLOCK_N = 512
    else:
        BLOCK_N = 1024
    
    # BLOCK_M should be large enough for good GPU occupancy
    BLOCK_M = 1024
    
    # Calculate grid dimensions
    grid_m = (n_elements + BLOCK_M * BLOCK_N - 1) // (BLOCK_M * BLOCK_N)
    grid_n = (hidden_size + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    layernorm_kernel[grid](
        x_ptr=x,
        gamma_ptr=gamma,
        beta_ptr=beta,
        out_ptr=out,
        n_elements=n_elements,
        hidden_size=hidden_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_layernorm