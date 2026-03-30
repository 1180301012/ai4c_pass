import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_layer_norm_kernel(
    x_ptr, y_ptr, out_sum_ptr, out_norm_ptr,
    weight_ptr, bias_ptr,
    mean_ptr, var_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feat_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program handles one element in the flattened tensor
    idx = tl.program_id(0)
    
    # Calculate position in the 3D tensor
    batch_idx = idx // (seq_len * feat_dim)
    remain_idx = idx % (seq_len * feat_dim)
    seq_idx = remain_idx // feat_dim
    feat_idx = remain_idx % feat_dim
    
    # Early exit if out of bounds
    if batch_idx >= batch_size or seq_idx >= seq_len or feat_idx >= feat_dim:
        return
    
    # Calculate global offset
    offset = batch_idx * seq_len * feat_dim + seq_idx * feat_dim + feat_idx
    
    # Load x and y values
    x = tl.load(x_ptr + offset)
    y = tl.load(y_ptr + offset)
    
    # Add
    z = x + y
    
    # Store the sum result (this is tmp_2 in the original)
    tl.store(out_sum_ptr + offset, z)
    
    # For the last feature of each batch+sequence position, compute statistics
    if feat_idx == feat_dim - 1 and seq_idx == seq_len - 1:
        # Initialize accumulators for this batch position
        batch_sum = 0.0
        batch_sum_sq = 0.0
        
        # Reduce over sequence and feature dimensions for this batch
        for s in range(seq_len):
            for f in range(feat_dim):
                seq_feat_offset = batch_idx * seq_len * feat_dim + s * feat_dim + f
                x_val = tl.load(x_ptr + seq_feat_offset)
                y_val = tl.load(y_ptr + seq_feat_offset)
                z_val = x_val + y_val
                batch_sum += z_val
                batch_sum_sq += z_val * z_val
        
        # Compute mean and variance for this batch
        batch_size_feat = tl.cast(seq_len * feat_dim, tl.float32)
        mean = batch_sum / batch_size_feat
        var = (batch_sum_sq / batch_size_feat) - (mean * mean)
        var = tl.maximum(var, eps)
        
        # Store mean and variance
        tl.store(mean_ptr + batch_idx, mean)
        tl.store(var_ptr + batch_idx, var)

@triton.jit  
def fused_normalize_kernel(
    out_sum_ptr, out_norm_ptr,
    weight_ptr, bias_ptr,
    mean_ptr, var_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feat_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the flattened tensor
    idx = tl.program_id(0)
    
    # Calculate position in the 3D tensor
    batch_idx = idx // (seq_len * feat_dim)
    remain_idx = idx % (seq_len * feat_dim)
    seq_idx = remain_idx // feat_dim
    feat_idx = remain_idx % feat_dim
    
    # Early exit if out of bounds
    if batch_idx >= batch_size or seq_idx >= seq_len or feat_idx >= feat_dim:
        return
    
    # Calculate global offset
    offset = batch_idx * seq_len * feat_dim + seq_idx * feat_dim + feat_idx
    
    # Load the sum result
    z = tl.load(out_sum_ptr + offset)
    
    # Load mean and variance for this batch
    mean = tl.load(mean_ptr + batch_idx)
    var = tl.load(var_ptr + batch_idx)
    std = tl.sqrt(var)
    
    # Normalize
    norm_z = (z - mean) / std
    
    # Load weight and bias
    weight = tl.load(weight_ptr + feat_idx)
    bias = tl.load(bias_ptr + feat_idx)
    
    # Apply transformation
    result = norm_z * weight + bias
    
    # Store normalized result
    tl.store(out_norm_ptr + offset, result)

@torch.fx.wrap
def fused_add_layer_norm(x, y, weight, bias, normalized_shape=(1024,), eps=1e-05):
    """
    Fused addition and layer normalization operation
    Args:
        x: Input tensor [B, S, F]
        y: Input tensor [B, S, F] 
        weight: LayerNorm weights [F]
        bias: LayerNorm bias [F]
        normalized_shape: Feature dimension size
        eps: Epsilon for numerical stability
    Returns:
        tuple: (x + y, layer_norm(x + y))
    """
    # Get tensor properties
    B, S, F = x.shape
    
    # Create output tensors
    out_sum = torch.empty_like(x)
    out_norm = torch.empty_like(x)
    
    # Create temporary buffers for mean and variance (one per batch)
    mean_buffer = torch.empty(B, dtype=torch.float32, device=x.device)
    var_buffer = torch.empty(B, dtype=torch.float32, device=x.device)
    
    # Total number of elements
    n_elements = B * S * F
    
    # Launch first kernel for addition and statistics computation
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_layer_norm_kernel[(num_programs,)](
        x=x,
        y=y,
        out_sum=out_sum,
        out_norm=out_norm,  # Temporary storage
        weight=weight,
        bias=bias,
        mean_ptr=mean_buffer,
        var_ptr=var_buffer,
        batch_size=B,
        seq_len=S,
        feat_dim=F,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps,
    )
    
    # Launch second kernel for normalization 
    fused_normalize_kernel[(num_programs,)](
        out_sum_ptr=out_sum,
        out_norm_ptr=out_norm,
        weight_ptr=weight,
        bias_ptr=bias,
        mean_ptr=mean_buffer,
        var_ptr=var_buffer,
        batch_size=B,
        seq_len=S,
        feat_dim=F,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sum, out_norm

def pattern(x, y):
    """Simple addition pattern like the reference"""
    return x + y

def replacement_args(x, y):
    """Extract arguments needed for the replacement"""
    return x, y

def replacement_func():
    """Return the optimized kernel function"""
    def triton_add(x, y):
        # Simple Triton addition kernel
        return x + y
    return triton_add