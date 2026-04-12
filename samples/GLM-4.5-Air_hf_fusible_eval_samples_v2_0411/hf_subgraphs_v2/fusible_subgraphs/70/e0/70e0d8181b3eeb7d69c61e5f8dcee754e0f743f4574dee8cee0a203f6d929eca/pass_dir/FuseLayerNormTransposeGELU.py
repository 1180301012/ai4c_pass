import torch
import triton
import triton.language as tl

# Pattern matching function - matches LayerNorm + Transpose + GELU sequence
def pattern(in_0, in_1, in_2):
    """Match LayerNorm + Transpose + GELU sequence"""
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for fused LayerNorm + Transpose + GELU
@triton.jit
def fused_layernorm_transpose_gelu_kernel(
    input_ptr,      # Input tensor [1, 3999, 512]
    weight_ptr,     # LayerNorm weight [512] 
    bias_ptr,       # LayerNorm bias [512]
    output_ptr,     # Output tensor [1, 512, 3999]
    n_batch: tl.constexpr,         # Batch size = 1
    n_seq: tl.constexpr,          # Sequence length = 3999
    n_features: tl.constexpr,     # Features = 512
    BLOCK_SIZE_SEQ: tl.constexpr,  # Block size for sequence dimension
    BLOCK_SIZE_FEAT: tl.constexpr, # Block size for feature dimension
    eps: tl.constexpr = 1e-5
):
    # Each program handles a block of features and sequence
    feat_idx = tl.program_id(0) * BLOCK_SIZE_FEAT + tl.arange(0, BLOCK_SIZE_FEAT)
    seq_idx = tl.program_id(1) * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    
    # Mask for valid indices
    feat_mask = feat_idx < n_features
    seq_mask = seq_idx < n_seq
    
    # Load weight and bias (broadcast across seq)
    weight = tl.load(weight_ptr + feat_idx, mask=feat_mask, other=0.0)
    bias = tl.load(bias_ptr + feat_idx, mask=feat_mask, other=0.0)
    
    # Load input data with transpose: [batch, seq, feat] -> [batch, feat, seq]
    # We need to transpose, so access as [batch, seq, feat] but store as [batch, feat, seq]
    input_offsets = feat_idx[:, None] + seq_idx[None, :] * n_features
    input_data = tl.load(input_ptr + input_offsets, mask=feat_mask[:, None] & seq_mask[None, :], other=0.0)
    
    # LayerNorm computation - simplified version
    # We compute LayerNorm along the feature dimension for each sequence position
    
    # Sum along features axis (axis=0) to get per-sequence means
    input_mean_tl = tl.sum(input_data, axis=0, dtype=tl.float32) / n_features
    
    # Broadcast mean to subtract from each feature value
    input_centered_tl = input_data - input_mean_tl
    
    # Compute variance along features
    variance_tl = tl.sum(input_centered_tl * input_centered_tl, axis=0, dtype=tl.float32) / n_features
    
    # Compute normalization factor with epsilon for numerical stability
    norm_factor_tl = tl.rsqrt(variance_tl + eps)
    
    # Apply LayerNorm normalization (center * scale + bias)
    layer_norm_out = input_centered_tl * norm_factor_tl[None, :] + bias[:, None]
    
    # Apply learnable weight and bias
    layer_norm_out = layer_norm_out * weight[:, None]
    
    # Apply GELU activation using a simple approximation that works in Triton
    # Using sigmoid for GELU approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
    x = layer_norm_out
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x * 1.702))
    gelu_out = x * sigmoid_x
    
    # Store result with transpose: [batch, feat, seq] layout
    output_offsets = seq_idx[:, None] + feat_idx[None, :] * n_seq
    tl.store(output_ptr + output_offsets, gelu_out, mask=feat_mask[None, :] & seq_mask[:, None])

# Kernel wrapper function (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_layernorm_transpose_gelu(in_0, in_1, in_2):
    # Get input dimensions
    n_batch, n_seq, n_features = in_2.shape
    
    # Create output tensor with transposed shape [1, 512, 3999]
    output = torch.empty((n_batch, n_features, n_seq), dtype=in_2.dtype, device=in_2.device)
    
    # Block sizes for optimal GPU utilization - use smaller blocks for better performance
    BLOCK_SIZE = 64   # Smaller block size for better GPU occupancy
    BLOCK_SIZE_FEAT = BLOCK_SIZE
    BLOCK_SIZE_SEQ = BLOCK_SIZE
    
    # Calculate grid dimensions
    grid_feat = (n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT
    grid_seq = (n_seq + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ
    
    # Launch Triton kernel
    fused_layernorm_transpose_gelu_kernel[(grid_feat, grid_seq)](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        n_batch=n_batch,
        n_seq=n_seq,
        n_features=n_features,
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ
    )
    
    return output

# Replacement function (returns function reference, no arguments)
def replacement_func():
    return fused_layernorm_transpose_gelu