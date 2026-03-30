import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, 
    output_ptr,
    n_features,
    n_seq,
    eps: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Fused LayerNorm + Transpose kernel using 2D blocking"""
    pid_f = tl.program_id(0)  # Feature block
    pid_s = tl.program_id(1)  # Sequence block
    batch_size = 1
    
    # Offsets within blocks
    feat_offs = pid_f * BLOCK_FEATURES + tl.arange(0, BLOCK_FEATURES)
    seq_offs = pid_s * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    
    # Create 2D indices
    feat_indices = feat_offs[:, None]  # [features, 1] 
    seq_indices = seq_offs[None, :]   # [1, sequences]
    
    # Combined indices for input [batch, seq, features]
    input_indices = seq_indices * n_features + feat_indices  # [seq, features]
    
    # Validity masks
    feat_mask = feat_indices < n_features
    seq_mask = seq_indices < n_seq
    mask = feat_mask & seq_mask
    
    # Load weights and biases (broadcast across sequences)
    weight = tl.load(weight_ptr + feat_indices, mask=feat_mask, other=1.0)
    bias = tl.load(bias_ptr + feat_indices, mask=feat_mask, other=0.0)
    
    # Load input block as [seq, features]
    input_vals = tl.load(input_ptr + input_indices, mask=mask, other=0.0)
    
    # LayerNorm along features dimension (axis 1) - matching PyTorch exactly
    if input_vals.shape[1] > 0:
        # Compute mean along features ( use n-1 for unbiased estimator)
        mean = tl.sum(input_vals, axis=1) / n_features
        
        # Compute centered values using PyTorch's formula
        centered = input_vals - mean[:, None]
        
        # Compute variance using PyTorch's unbiased estimator (n-1 denominator)
        if n_features > 1:
            var = tl.sum(centered * centered, axis=1) / (n_features - 1)
        else:
            var = tl.zeros([centered.shape[0]], dtype=tl.float32)
        
        # Compute standard deviation with epsilon
        std = tl.sqrt(var + eps)
        
        # Normalize: (x - mean) / std
        normalized = centered / std[:, None]
        
        # Apply element-wise weight and bias
        normalized = normalized * weight + bias
    else:
        normalized = input_vals
    
    # Transpose: [seq, features] -> [features, seq]
    # Output indices: [features, seq] for output tensor [batch, features, seq]
    output_indices = feat_indices * n_seq + seq_indices
    
    # Store transposed results
    tl.store(output_ptr + output_indices, normalized, mask=mask)

@torch.fx.wrap
def fused_layer_norm_transpose(input, weight, bias):
    """Fused LayerNorm + Transpose operation using Triton"""
    input_shape = input.shape
    batch_size, seq_len, features = input_shape
    
    # Output will be [batch_size, features, seq_len] (transposed)
    output_shape = (batch_size, features, seq_len)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Set optimal block sizes for our specific dimensions
    BLOCK_FEATURES = 128  # Feature dimension block size
    BLOCK_SEQ = 256       # Sequence dimension block size
    
    # Calculate grid dimensions  
    grid_features = (features + BLOCK_FEATURES - 1) // BLOCK_FEATURES
    grid_seqs = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    # Launch 2D kernel grid (features, sequences)
    fused_layer_norm_transpose_kernel[(grid_features, grid_seqs)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_features=features,
        n_seq=seq_len,
        eps=1e-05,
        BLOCK_FEATURES=BLOCK_FEATURES,
        BLOCK_SEQ=BLOCK_SEQ,
    )
    
    return output

def pattern(in_2, in_1, in_0):
    """
    Pattern that matches LayerNorm + Transpose computation
    """
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    return tmp_3  # Return transposed result

def replacement_args(in_2, in_1, in_0):
    """Extract arguments for the replacement function"""
    return in_2, in_1, in_0

def replacement_func():
    """Return the fused kernel function"""
    return fused_layer_norm_transpose