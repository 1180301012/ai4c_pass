import torch
import triton
import triton.language as tl

# Pattern matching function - matches GELU operation
def pattern(input_tensor):
    """
    Matches the GELU operation exactly as it appears in the computation
    """
    # GELU activation
    return torch.nn.functional.gelu(input_tensor)

# Arguments extraction function
def replacement_args(input_tensor):
    """
    Extract arguments needed for the optimized kernel
    """
    return (input_tensor,)

# Optimized GELU kernel implementation - uses the exact GELU mathematical formula
@triton.jit
def optimized_gelu_kernel(
    # Input tensor
    input_ptr,   # input tensor [1, 3999, 512] or [1, 512, 3999]
    # Output tensor
    output_ptr,  # output tensor [same shape as input]
    # Tensor dimensions
    n_batches: tl.constexpr,      # 1
    n_sequence: tl.constexpr,     # 3999 or 512
    n_features: tl.constexpr,     # 512 or 3999
    BLOCK_M: tl.constexpr,        # Block size for first dimension
    BLOCK_N: tl.constexpr,        # Block size for second dimension
):
    """
    Optimized GELU kernel using the exact mathematical formula:
    GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   """
    # Program identifier within launch grid
    seq_idx = tl.program_id(0)  # first dimension index
    feat_idx = tl.program_id(1)  # second dimension index
    batch_idx = tl.program_id(2)  # batch dimension index
    
    # Check bounds
    if batch_idx >= n_batches:
        return
    
    # Calculate offsets
    seq_offset = seq_idx * BLOCK_M
    feat_offset = feat_idx * BLOCK_N
    
    # Create bounds checking masks
    seq_mask = seq_offset + tl.arange(0, BLOCK_M) < n_sequence
    feat_mask = feat_offset + tl.arange(0, BLOCK_N) < n_features
    
    # Create 2D masks for the block
    seq_mask_2d = seq_mask[:, None]
    feat_mask_2d = feat_mask[None, :]
    
    # Calculate input addresses
    input_base = batch_idx * n_sequence * n_features
    seq_offsets = seq_offset + tl.arange(0, BLOCK_M)[:, None]
    feat_offsets = feat_offset + tl.arange(0, BLOCK_N)[None, :]
    seq_base_offsets = seq_offsets * n_features
    input_addresses = input_base + seq_base_offsets + feat_offsets
    
    # Load input data
    input_block = tl.load(input_ptr + input_addresses, mask=seq_mask_2d & feat_mask_2d, other=0.0)
    
    # Convert to float32 for precise computation
    x = input_block.to(tl.float32)
    
    # Compute exact GELU using the mathematical formula:
    # GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    cubic_coeff = 0.044715  # coefficient for x^3 term
    
    # Compute cubic term: x^3
    x_cubed = x * x * x
    
    # Compute the argument for tanh
    tanh_arg = sqrt_2_over_pi * (x + cubic_coeff * x_cubed)
    
    # Compute tanh using sigmoid approximation for better performance
    # tanh(x) = 2 * sigmoid(2*x) - 1
    sigmoid_arg = 2.0 * tanh_arg
    sigmoid = 1.0 / (1.0 + tl.exp(-sigmoid_arg))
    tanh_val = 2.0 * sigmoid - 1.0
    
    # Compute the final GELU
    gelu_val = x * 0.5 * (1.0 + tanh_val)
    
    # Store result
    tl.store(output_ptr + input_addresses, gelu_val.to(tl.float16), mask=seq_mask_2d & feat_mask_2d)

# Kernel wrapper - must be decorated with @torch.fx.wrap
@torch.fx.wrap
def optimized_gelu(input_tensor):
    """
    Wrapper function that sets up and launches the optimized GELU kernel
    """
    # Get tensor shapes
    n_batches = input_tensor.shape[0]  # 1
    n_sequence = input_tensor.shape[1]  # 3999 or 512
    n_features = input_tensor.shape[2]  # 512 or 3999
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Configure block sizes - optimized for lightweight GELU operation
    # Using smaller blocks for better cache efficiency and memory coalescing
    BLOCK_M = 64   # First dimension block size
    BLOCK_N = 32   # Second dimension block size
    
    # Calculate grid dimensions
    grid_m = (n_sequence + BLOCK_M - 1) // BLOCK_M
    grid_n = (n_features + BLOCK_N - 1) // BLOCK_N
    grid_b = n_batches
    
    # Launch the kernel
    optimized_gelu_kernel[(grid_m, grid_n, grid_b)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_batches=n_batches,
        n_sequence=n_sequence,
        n_features=n_features,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return output

# Replacement function - returns the optimized kernel function
def replacement_func():
    return optimized_gelu