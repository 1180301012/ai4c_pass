import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0, in_1, in_2):
    """
    Matches the sequence: LayerNorm -> Transpose -> GELU
    This mirrors the exact operations in model.py
    """
    # LayerNorm with parameters
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    # Transpose last two dimensions
    tmp_3 = tmp_2.transpose(-2, -1)
    # GELU activation
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4

# Arguments extraction function
def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the optimized kernel
    Returns the three input tensors for the fused operation
    """
    return (in_0, in_1, in_2)

# Optimized kernel implementation
@triton.jit
def fused_layernorm_transpose_gelu_kernel(
    # Input tensors
    bias_ptr,  # in_0: bias [512]
    weight_ptr,  # in_1: weight [512] 
    input_ptr,  # in_2: input [1, 3999, 512]
    # Output tensor  
    output_ptr,  # output [1, 512, 3999]
    # Tensor dimensions
    n_batches,      # 1
    n_sequence,     # 3999
    n_features,     # 512
    # Constants
    eps: tl.constexpr,  # 1e-05
    BLOCK_M: tl.constexpr,  # Block size for sequence dimension
    BLOCK_N: tl.constexpr,  # Block size for feature dimension
):
    """
    Fused kernel that performs:
    1. LayerNorm on the feature dimension (512)  
    2. Transpose of last two dimensions (3999, 512) -> (512, 3999)
    3. GELU activation
    
    Optimized for tensor shape [1, 3999, 512] -> [1, 512, 3999]
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
    
    # Load weight and bias vectors
    # Only load the portion we need for this block
    weight_vec = tl.load(weight_ptr + feat_offset + tl.arange(0, BLOCK_N), mask=feat_mask, other=1.0)
    bias_vec = tl.load(bias_ptr + feat_offset + tl.arange(0, BLOCK_N), mask=feat_mask, other=0.0)
    
    # Load input block: [block_sequence, block_features]
    # Input layout: [batch, sequence, features]
    seq_offsets = seq_offset + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
    feat_offsets = feat_offset + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]
    
    # Calculate input addresses: batch * seq * feat + seq * feat + feat
    input_base = batch_idx * n_sequence * n_features
    seq_base_offsets = seq_offsets * n_features
    input_addresses = input_base + seq_base_offsets + feat_offsets
    
    # Load input data with proper bounds checking
    input_block = tl.load(input_ptr + input_addresses, mask=seq_mask_2d & feat_mask_2d, other=0.0)
    
    # Convert to float32 for computation
    input_block_fp32 = input_block.to(tl.float32)
    
    # Compute mean and variance along the feature dimension for each sequence element
    # Use sum and manual division since tl.mean is not available
    input_squared = input_block_fp32 ** 2
    mean = tl.sum(input_block_fp32, axis=1) * (1.0 / BLOCK_N)
    mean_sq = tl.sum(input_squared, axis=1) * (1.0 / BLOCK_N)
    variance = mean_sq - mean * mean
    
    # Add epsilon for numerical stability
    std = tl.sqrt(variance + eps)
    
    # LayerNorm: normalize and apply weight/bias
    normalized = (input_block_fp32 - mean[:, None]) / std[:, None]
    layernorm_out = normalized * weight_vec + bias_vec
    
    # Apply GELU activation using sigmoid approximation
    x = layernorm_out
    gelu_arg = 0.7071067811865476 * x  # x / sqrt(2)
    sigmoid = 1.0 / (1.0 + tl.exp(-gelu_arg))
    gelu_out = x * sigmoid
    
    # Store output with transpose: swap sequence and feature dimensions
    # Output layout: [batch, features, sequence] 
    # We need to map (seq, feat) -> (feat, seq)
    
    # Calculate output addresses with transpose
    output_base = batch_idx * n_features * n_sequence
    feat_out_base = feat_offset + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]
    seq_out_base = seq_offset + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
    
    # Transposed addresses: feat * seq + seq
    transposed_seq_offsets = feat_out_base * n_sequence + seq_out_base
    
    output_addresses = output_base + transposed_seq_offsets
    
    # Store the result with bounds checking
    tl.store(output_ptr + output_addresses, gelu_out, mask=seq_mask_2d & feat_mask_2d)

# Kernel wrapper - must be decorated with @torch.fx.wrap
@torch.fx.wrap
def fused_layernorm_transpose_gelu(in_0, in_1, in_2):
    """
    Wrapper function that sets up and launches the fused kernel
    """
    # Get tensor shapes
    n_batches = in_2.shape[0]  # 1
    n_sequence = in_2.shape[1]  # 3999
    n_features = in_2.shape[2]  # 512
    
    # Create output tensor with transposed dimensions
    output_shape = (n_batches, n_features, n_sequence)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Configure block sizes based on tensor characteristics
    # Use smaller blocks for better memory coalescing with large sequence length
    BLOCK_M = 16  # sequence dimension block size
    BLOCK_N = 32  # feature dimension block size
    
    # Calculate grid dimensions
    grid_m = (n_sequence + BLOCK_M - 1) // BLOCK_M
    grid_n = (n_features + BLOCK_N - 1) // BLOCK_N
    grid_b = n_batches
    
    # Launch the kernel
    fused_layernorm_transpose_gelu_kernel[(grid_m, grid_n, grid_b)](
        bias_ptr=in_0,
        weight_ptr=in_1, 
        input_ptr=in_2,
        output_ptr=output,
        n_batches=n_batches,
        n_sequence=n_sequence,
        n_features=n_features,
        eps=1e-05,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return output

# Replacement function - returns the optimized kernel function
def replacement_func():
    return fused_layernorm_transpose_gelu