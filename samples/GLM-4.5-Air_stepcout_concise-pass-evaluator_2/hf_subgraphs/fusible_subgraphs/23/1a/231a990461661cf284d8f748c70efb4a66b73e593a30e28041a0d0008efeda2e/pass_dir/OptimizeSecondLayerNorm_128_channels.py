import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    n_features,
    n_elements,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    row_offset = row_idx * n_features
    
    # Create a mask for the current row
    mask = tl.arange(0, BLOCK_SIZE) < n_features
    
    # Load the current row
    x_row = tl.load(x_ptr + row_offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Load gamma and beta for the current features
    gamma_row = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    beta_row = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Compute mean
    x_mean = tl.sum(x_row, axis=0) / n_features
    
    # Compute variance
    x_var = tl.sum((x_row - x_mean) ** 2, axis=0) / n_features
    
    # Compute normalized output
    x_norm = (x_row - x_mean) / tl.sqrt(x_var + eps)
    output_row = x_norm * gamma_row + beta_row
    
    # Store the result
    tl.store(output_ptr + row_offset + tl.arange(0, BLOCK_SIZE), output_row, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, eps=1e-05):
    # Get input dimensions
    if x.dim() == 3:
        batch_size, n_patches, n_features = x.shape
    else:
        raise ValueError("Input must be 3D tensor for this optimized layer norm")
    
    # Reshape to [batch_size * n_patches, n_features] for processing
    x_flat = x.reshape(-1, n_features)
    output_flat = torch.empty_like(x_flat)
    
    # Calculate number of programs needed
    n_rows = x_flat.shape[0]
    BLOCK_SIZE = min(1024, n_features)
    n_programs = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    layer_norm_kernel[(n_programs,)](
        x_ptr=x_flat,
        gamma_ptr=weight,
        beta_ptr=bias,
        output_ptr=output_flat,
        n_features=n_features,
        n_elements=0,  # Not used in this kernel
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original dimensions
    return output_flat.reshape(batch_size, n_patches, n_features)

# Pattern matching function - matches second LayerNorm operation
def pattern(tmp_11, tmp_6, tmp_5):
    tmp_12 = torch.nn.functional.layer_norm(tmp_11, (128,), tmp_6, tmp_5, 1e-05)
    return tmp_12

# Argument extraction function  
def replacement_args(tmp_11, tmp_6, tmp_5):
    return (tmp_11, tmp_6, tmp_5)

# Replacement function
def replacement_func():
    return optimized_layer_norm