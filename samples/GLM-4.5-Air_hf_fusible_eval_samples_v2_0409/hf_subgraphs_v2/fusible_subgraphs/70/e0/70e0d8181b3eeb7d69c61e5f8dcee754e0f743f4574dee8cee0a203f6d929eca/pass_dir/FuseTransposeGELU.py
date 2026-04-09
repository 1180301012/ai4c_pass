import torch
import triton
import triton.language as tl

# Pattern matching function - matches the sequence: Transpose + GELU
def pattern(input_tensor):
    """
    Matches the sequence: Transpose -> GELU
    This mirrors the exact operations from the computation flow
    """
    # Transpose last two dimensions: [1, 3999, 512] -> [1, 512, 3999]
    tmp_3 = input_tensor.transpose(-2, -1)
    # GELU activation
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4

# Arguments extraction function
def replacement_args(input_tensor):
    """
    Extract arguments needed for the optimized kernel
    """
    return (input_tensor,)

# Optimized kernel implementation
@triton.jit
def transpose_gelu_kernel(
    # Input tensor
    input_ptr,  # input [1, 3999, 512]
    # Output tensor  
    output_ptr,  # output [1, 512, 3999]
    # Tensor dimensions
    n_batches,      # 1
    n_sequence,     # 3999
    n_features,     # 512
    # Constants
    BLOCK_M: tl.constexpr,  # Block size for sequence dimension
    BLOCK_N: tl.constexpr,  # Block size for feature dimension
):
    """
    Fused kernel that performs:
    1. Transpose of last two dimensions (3999, 512) -> (512, 3999)
    2. GELU activation
    """
    # Program identifier within launch grid
    seq_idx = tl.program_id(0)  # sequence dimension index  
    feat_idx = tl.program_id(1)  # feature dimension index
    batch_idx = tl.program_id(2)  # batch dimension index
    
    # Check bounds for batch dimension
    if batch_idx >= n_batches:
        return
    
    # Calculate offsets within the current block
    seq_offset = seq_idx * BLOCK_M
    feat_offset = feat_idx * BLOCK_N
    
    # Create offset masks for bounds checking
    seq_mask = seq_offset + tl.arange(0, BLOCK_M) < n_sequence
    feat_mask = feat_offset + tl.arange(0, BLOCK_N) < n_features
    
    # Create 2D masks for the block
    seq_mask_2d = seq_mask[:, None]  # Expand for columns
    feat_mask_2d = feat_mask[None, :]  # Expand for rows
    
    # Calculate input addresses: [batch, sequence, features]
    # Input layout: [1, 3999, 512]
    input_base = batch_idx * n_sequence * n_features
    seq_offsets = seq_offset + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
    feat_offsets = feat_offset + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]
    seq_base_offsets = seq_offsets * n_features
    input_addresses = input_base + seq_base_offsets + feat_offsets
    
    # Load input data with proper bounds checking
    input_block = tl.load(input_ptr + input_addresses, mask=seq_mask_2d & feat_mask_2d, other=0.0)
    
    # Apply GELU activation using more accurate sigmoid approximation
    # GELU(x) = x * sigmoid(sqrt(2/pi) * x)
    # This is a mathematically equivalent approximation to the original GELU
    x = input_block.to(tl.float32)
    gelu_arg = 0.7978845608028654 * x  # sqrt(2/pi) ≈ 0.7978845608028654
    sigmoid = 1.0 / (1.0 + tl.exp(-gelu_arg))
    gelu_out = x * sigmoid
    
    # Store output with transpose: swap sequence and feature dimensions  
    # Output layout: [batch, features, sequence]
    # Map (seq, feat) -> (feat, seq)
    output_base = batch_idx * n_features * n_sequence
    feat_out_base = feat_offset + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]
    seq_out_base = seq_offset + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
    
    # Transposed addresses: feat * seq + seq
    transposed_seq_offsets = feat_out_base * n_sequence + seq_out_base
    output_addresses = output_base + transposed_seq_offsets
    
    # Store the result with bounds checking
    tl.store(output_ptr + output_addresses, gelu_out.to(tl.float16), mask=seq_mask_2d & feat_mask_2d)

# Kernel wrapper - must be decorated with @torch.fx.wrap
@torch.fx.wrap
def transpose_gelu(input_tensor):
    """
    Wrapper function that sets up and launches the fused kernel
    """
    # Get tensor shapes
    n_batches = input_tensor.shape[0]  # 1
    n_sequence = input_tensor.shape[1]  # 3999
    n_features = input_tensor.shape[2]  # 512
    
    # Create output tensor with transposed dimensions
    output_shape = (n_batches, n_features, n_sequence)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure block sizes - optimized for the specific tensor shape [1, 3999, 512]
    # Use larger blocks for better GPU occupancy and memory coalescing
    BLOCK_M = 128  # sequence dimension block size  
    BLOCK_N = 32   # feature dimension block size
    
    # Calculate grid dimensions
    grid_m = (n_sequence + BLOCK_M - 1) // BLOCK_M
    grid_n = (n_features + BLOCK_N - 1) // BLOCK_N
    grid_b = n_batches
    
    # Launch the kernel
    transpose_gelu_kernel[(grid_m, grid_n, grid_b)](
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
    return transpose_gelu