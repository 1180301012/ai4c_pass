import torch
import triton
import triton.language as tl

def pattern(x, weight):
    """Pattern: torch.nn.functional.linear(x, weight, None)"""
    return torch.nn.functional.linear(x, weight, None)

def replacement_args(x, weight):
    return (x, weight)

@triton.jit
def linear_kernel(
    x_ptr, 
    weight_ptr, 
    out_ptr,
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
):
    """Simple linear transformation kernel using Triton"""
    pid = tl.program_id(0)
    
    # Each program handles one output element
    # Calculate which output element this program handles
    output_idx = pid
    
    # Convert to batch and feature indices
    batch_idx = output_idx // out_features
    feature_idx = output_idx % out_features
    
    # Initialize output
    output = 0.0
    
    # Compute dot product: x[batch_idx, :] @ weight[feature_idx, :]
    for k in range(in_features):
        # Load x element with bounds checking
        x_mask = (batch_idx < batch_size) & (k < in_features)
        x_val = tl.load(x_ptr + batch_idx * in_features + k, mask=x_mask, other=0.0)
        
        # Load weight element with bounds checking
        w_mask = (feature_idx < out_features) & (k < in_features) 
        w_val = tl.load(weight_ptr + feature_idx * in_features + k, mask=w_mask, other=0.0)
        
        output += x_val * w_val
    
    # Store result with bounds checking
    mask = (output_idx < batch_size * out_features)
    tl.store(out_ptr + output_idx, output, mask=mask)

@torch.fx.wrap
def optimized_linear(x, weight):
    """Optimized linear transformation using Triton"""
    # Get original shape information
    original_shape = x.shape
    if len(original_shape) == 3:
        batch_size, seq_len, in_features = original_shape
        total_elements = batch_size * seq_len
    elif len(original_shape) == 2:
        total_elements, in_features = original_shape
        batch_size = total_elements
        seq_len = 1
    else:
        # 1D tensor - treat as single batch
        total_elements = 1
        in_features = original_shape[0] if len(original_shape) == 1 else 1
        batch_size = 1
        seq_len = 1
    
    out_features = weight.shape[0]
    
    total_output_elements = batch_size * out_features
    
    # Calculate number of programs needed (one per output element)
    num_programs = total_output_elements
    
    # Allocate output tensor in flattened form
    out_flat = torch.empty(total_output_elements, dtype=x.dtype, device=x.device)
    
    # Launch simplified kernel
    if num_programs > 0:
        linear_kernel[(num_programs,)](
            x_ptr=x,
            weight_ptr=weight,
            out_ptr=out_flat,
            batch_size=batch_size,
            in_features=in_features,
            out_features=out_features,
        )
    
    # Return flattened result since view() is not allowed
    return out_flat

def replacement_func():
    return optimized_linear