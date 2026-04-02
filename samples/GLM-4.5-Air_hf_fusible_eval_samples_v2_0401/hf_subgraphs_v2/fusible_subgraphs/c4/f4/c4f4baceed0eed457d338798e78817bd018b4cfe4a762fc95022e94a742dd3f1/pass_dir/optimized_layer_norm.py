import torch
import triton
import triton.language as tl

def pattern(input_tensor, normalized_shape, weight_tensor, bias_tensor, eps):
    # Match layer norm operation as it appears in the actual computation
    result = torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight_tensor, bias_tensor, eps)
    return result

def replacement_args(input_tensor, normalized_shape, weight_tensor, bias_tensor, eps):
    return (input_tensor, normalized_shape, weight_tensor, bias_tensor, eps)

@triton.jit
def triton_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    feature_dim,
    eps: tl.constexpr,
):
    # Each program handles a sequence position across ALL feature dimensions
    pid = tl.program_id(0)
    
    # Calculate batch and sequence position (assuming [batch_size, seq_len, features])
    seq_len = n_elements // feature_dim
    if pid >= seq_len:
        return  # Handle boundary cases
    
    # Compute mean along feature dimension for this sequence position
    mean_val = 0.0
    for feature_idx in range(feature_dim):
        input_val = tl.load(
            input_ptr + pid * feature_dim + feature_idx,
            mask=feature_idx < feature_dim,
            other=0.0
        ).to(tl.float32)
        mean_val += input_val
    mean_val /= feature_dim
    
    # Compute variance along feature dimension for this sequence position
    var_val = 0.0
    for feature_idx in range(feature_dim):
        input_val = tl.load(
            input_ptr + pid * feature_dim + feature_idx,
            mask=feature_idx < feature_dim,
            other=0.0
        ).to(tl.float32)
        var_val += (input_val - mean_val) * (input_val - mean_val)
    var_val /= feature_dim
    
    # Apply normalization, weight, and bias for each feature
    for feature_idx in range(feature_dim):
        input_val = tl.load(
            input_ptr + pid * feature_dim + feature_idx,
            mask=feature_idx < feature_dim,
            other=0.0
        ) .to(tl.float32)
        
        weight_val = tl.load(
            weight_ptr + feature_idx,
            mask=feature_idx < feature_dim,
            other=1.0
        ).to(tl.float32)
        
        bias_val = tl.load(
            bias_ptr + feature_idx,
            mask=feature_idx < feature_dim,
            other=0.0
        ).to(tl.float32)
        
        # Layer norm formula: (x - mean) / sqrt(var + eps) * weight + bias
        inv_std = 1.0 / tl.sqrt(var_val + eps)
        normalized_val = (input_val - mean_val) * inv_std * weight_val + bias_val
        
        # Store output (normalized_val is already in float32, Triton will handle conversion)
        tl.store(
            output_ptr + pid * feature_dim + feature_idx,
            normalized_val.to(tl.float32),
            mask=feature_idx < feature_dim
        )

@torch.fx.wrap  
def triton_layer_norm(input_tensor, normalized_shape, weight_tensor, bias_tensor, eps):
    # Get tensor information
    input_shape = input_tensor.shape
    batch_size, seq_len, feature_dim = input_shape
    
    # Reshape input to [batch_size * seq_len, feature_dim] for processing
    input_2d = input_tensor.reshape(-1, feature_dim)
    n_elements = input_2d.numel()
    
    # Create output tensor
    output_2d = torch.empty_like(input_2d)
    
    # Launch kernel - one program per sequence position
    grid_size = (batch_size * seq_len,)
    
    triton_layer_norm_kernel[grid_size](
        input_ptr=input_2d,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=output_2d,
        n_elements=n_elements,
        feature_dim=feature_dim,
        eps=eps
    )
    
    # Reshape back to original 3D format
    return output_2d.reshape(input_shape)

def replacement_func():
    return triton_layer_norm