import torch
import triton
import triton.language as tl

def pattern(bias, weight, x):
    """Pattern matching for layer_norm followed by transpose"""
    # Layer norm with normalized_shape (768,), weight, bias, eps=1e-05
    tmp_2 = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)
    # Transpose result along last two dimensions
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

def replacement_args(bias, weight, x):
    """Extract arguments for the fused kernel"""
    return (bias, weight, x)

@triton.jit
def layer_norm_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    features,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """Layer norm kernel"""
    # Get program IDs
    pid = tl.program_id(0)
    batch_id = pid // ((seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    seq_id = (pid % ((seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)) * BLOCK_SIZE_M
    
    # Compute offsets
    batch_offsets = batch_id
    seq_offsets = seq_id + tl.arange(0, BLOCK_SIZE_M)
    feat_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    batch_mask = batch_offsets < batch_size
    seq_mask = seq_offsets < seq_len
    feat_mask = feat_offsets < features
    
    # Load weights and bias
    weight = tl.load(weight_ptr + feat_offsets, mask=feat_mask)
    bias = tl.load(bias_ptr + feat_offsets, mask=feat_mask)
    
    # Load input data: [batch_id, seq_offsets, feat_offsets]
    batch_stride = seq_len * features
    x_ptrs = x_ptr + batch_offsets * batch_stride + seq_offsets[:, None] * features + feat_offsets[None, :]
    x = tl.load(x_ptrs, mask=batch_mask[:, None] * seq_mask[:, None] * feat_mask[None, :])
    
    # Compute mean and variance for layer norm along features dimension (axis=1) 
    x_mean = tl.sum(x, axis=1) / tl.sum(feat_mask).to(tl.float32)
    x_var = tl.sum((x - x_mean[:, None]) * (x - x_mean[:, None]), axis=1) / tl.sum(feat_mask).to(tl.float32)
    
    # Normalize along features dimension
    x_norm = (x - x_mean[:, None]) / tl.sqrt(x_var[:, None] + eps)
    
    # Scale and shift
    x_scaled = x_norm * weight + bias
    
    # Store result - same shape as input for now
    out_ptrs = x_ptrs  # Same memory layout
    tl.store(out_ptrs, x_scaled, mask=batch_mask[:, None] * seq_mask[:, None] * feat_mask[None, :])

@torch.fx.wrap
def fused_layer_norm_transpose(bias, weight, x):
    """Wrapper for the layer norm kernel"""
    # Get input dimensions
    batch_size, seq_len, features = x.shape
    
    # Set block sizes - tuned for layer_norm workload
    BLOCK_SIZE_M = 32   # Process smaller seq chunks for better memory locality
    BLOCK_SIZE_N = 128  # Smaller feature chunk size for better cache utilization
    
    # Calculate grid size
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_size = batch_size * grid_m
    grid = (grid_size,)
    
    # Create output tensor (same shape as input for now)
    out = torch.empty_like(x)
    
    # Launch kernel
    layer_norm_kernel[grid](
        bias,
        weight, 
        x,
        out,
        batch_size,
        seq_len,
        features,
        eps=1e-05,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Now do the transpose operation separately
    return out.transpose(-1, -2)

def replacement_func():
    """Return the fused function"""
    return fused_layer_norm_transpose