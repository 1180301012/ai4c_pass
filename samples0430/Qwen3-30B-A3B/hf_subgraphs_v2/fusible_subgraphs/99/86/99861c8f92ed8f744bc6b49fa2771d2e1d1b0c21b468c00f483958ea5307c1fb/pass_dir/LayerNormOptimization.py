import torch
import triton
import triton.language as tl

# Pattern matching: matches layer_norm with normalized_shape (1024,)
def pattern(x, norm_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, norm_shape, weight, bias, eps)

def replacement_args(x, norm_shape, weight, bias, eps):
    # Verify normalized_shape is (1024,) for our optimization
    if norm_shape != (1024,):
        return None
    return (x, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    seq_len, feature_size,
    eps: tl.float32
):
    # One block per sequence position
    seq_id = tl.program_id(0)
    
    # Offset calculation for this sequence
    x_row_offset = seq_id * feature_size
    out_row_offset = seq_id * feature_size
    
    # Load full row (features) from input
    x = tl.load(
        x_ptr + x_row_offset,
        mask=tl.arange(0, feature_size) < feature_size,
        other=0.0
    )
    
    # Compute mean across features
    mean = tl.sum(x, axis=0) / tl.float32(feature_size)
    
    # Compute variance
    x_minus_mean = x - mean
    var = tl.sum(x_minus_mean * x_minus_mean, axis=0) / tl.float32(feature_size)
    inv_var = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    normalized = x_minus_mean * inv_var
    
    # Load weight and bias vectors
    weight = tl.load(
        weight_ptr,
        mask=tl.arange(0, feature_size) < feature_size,
        other=0.0
    )
    bias = tl.load(
        bias_ptr,
        mask=tl.arange(0, feature_size) < feature_size,
        other=0.0
    )
    
    # Apply scale and shift
    normalized = normalized * weight + bias
    
    # Store result
    tl.store(
        out_ptr + out_row_offset,
        normalized,
        mask=tl.arange(0, feature_size) < feature_size
    )

@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias, eps):
    # Extract tensor shapes: [batch, seq_len, feature_size]
    B, seq_len, feature_size = x.shape
    out = torch.empty_like(x)
    
    # Launch kernel: one block per sequence position
    num_programs = seq_len
    
    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        seq_len=seq_len,
        feature_size=feature_size,
        eps=eps
    )
    
    return out

def replacement_func():
    return layer_norm_wrapper