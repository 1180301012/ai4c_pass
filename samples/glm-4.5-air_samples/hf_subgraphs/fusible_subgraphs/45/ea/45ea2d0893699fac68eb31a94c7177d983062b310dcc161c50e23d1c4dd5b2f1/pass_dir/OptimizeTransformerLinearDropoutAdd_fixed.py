import torch
import triton
import triton.language as tl

def pattern(gm, example_inputs):
    return True

def replacement_args(bias, weight, x, addend):
    return (bias, weight, x, addend)

@triton.jit
def fused_linear_dropout_add_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    addend_ptr,
    out_ptr,
    batch_size,
    in_features,
    out_features,
    p_dropout,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element per batch
    batch_idx = tl.program_id(0) // out_features
    feature_idx = tl.program_id(0) % out_features
    
    if batch_idx >= batch_size or feature_idx >= out_features:
        return
    
    # Compute linear result: bias + sum(x @ weight)
    linear_result = bias_ptr[feature_idx]
    
    # Vectorized dot product
    for k in range(0, in_features, BLOCK_SIZE):
        # Load chunk of input and weights
        k_end = min(k + BLOCK_SIZE, in_features)
        mask = tl.arange(k, k_end) < in_features
        
        # Load x and weight vectors for this computation
        x_vals = tl.load(x_ptr + batch_idx * in_features + tl.arange(k, k_end), mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + feature_idx * in_features + tl.arange(k, k_end), mask=mask, other=0.0)
        
        # Accumulate dot product
        linear_result += tl.sum(x_vals * weight_vals)
    
    # Fast dropout implementation for training=False (fixed random pattern)
    # Generate random numbers using offset-based deterministic pattern
    r = tl.rand(batch_idx * out_features + feature_idx)  # Returns float in [0, 1)
    
    # Apply dropout: keep probability = 1 - p_dropout
    scale = 1.0 / (1.0 - p_dropout)  # Scale to maintain expected value
    mask_keep = r > p_dropout
    dropout_result = linear_result * mask_keep * scale
    
    # Load addend for this specific output element
    addend_idx = batch_idx * out_features + feature_idx
    addend_val = tl.load(addend_ptr + addend_idx)
    
    # Fused addition
    out_result = addend_val + dropout_result
    
    # Store result
    tl.store(out_ptr + addend_idx, out_result)

@torch.fx.wrap
def kernel_wrapper(bias, weight, x, addend):
    # Handle different tensor dimensions
    if x.dim() == 2:
        # Shape [batch, features]
        batch_size, in_features = x.shape
        out_features = bias.shape[0]
    else:
        # Handle 3D tensors by flattening spatial dimensions
        batch_size = x.shape[0]
        spatial_size = x.shape[1] * x.shape[2] if x.dim() == 3 else 1
        in_features = x.shape[-1] if x.dim() == 3 else spatial_size
        out_features = bias.shape[0]
    
    # Reshape x to 2D for easier computation
    if x.dim() == 3:
        x_2d = x.reshape(-1, in_features)
    else:
        x_2d = x
    
    # Create output tensor with appropriate shape
    if x.dim() == 3:
        out_shape = x.shape
    else:
        out_shape = (batch_size, out_features)
    
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Reshape output for 2D computation
    if out.dim() == 3:
        out_2d = out.reshape(-1, out_features)
    else:
        out_2d = out
    
    # Reshape addend if needed
    if addend.dim() == 3 and addend.shape != x.shape:
        addend_2d = addend.reshape(-1, out_features)
    else:
        addend_2d = addend.reshape(-1, out_features) if addend.dim() != 2 else addend
    
    # Dropout probability
    p_dropout = 0.1
    
    # Determine optimal block size
    BLOCK_SIZE = 32
    
    # Launch fused kernel
    total_elements = batch_size * out_features
    grid_size = total_elements
    
    fused_linear_dropout_add_kernel[grid_size](
        x_ptr=x_2d,
        weight_ptr=weight,
        bias_ptr=bias,
        addend_ptr=addend_2d,
        out_ptr=out_2d,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        p_dropout=p_dropout,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return kernel_wrapper