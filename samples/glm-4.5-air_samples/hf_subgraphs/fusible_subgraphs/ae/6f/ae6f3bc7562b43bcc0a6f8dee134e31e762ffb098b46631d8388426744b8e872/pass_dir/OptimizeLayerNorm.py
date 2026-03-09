import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Pattern: type conversion + layer normalization
    tmp_10 = x.to(torch.float32)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1280,), weight, bias, 1e-12)
    return tmp_11

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    n_elements,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row of the input tensor
    row_offset = tl.program_id(0) * feat_dim
    
    # Load one row of features
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_offset + feat_dim
    
    # Load input, gamma, and beta
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float32 if needed (this is handled by calling function)
    x_f32 = tl.cast(x, tl.float32)
    
    # Compute mean
    x_mean = tl.sum(x_f32, axis=0) / feat_dim
    x_var = tl.sum((x_f32 - x_mean) * (x_f32 - x_mean), axis=0) / feat_dim
    
    # Apply layer normalization
    eps = 1e-12
    x_norm = (x_f32 - x_mean) / tl.sqrt(x_var + eps)
    out = x_norm * gamma + beta
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layernorm(x, weight, bias):
    # Get shapes
    if len(x.shape) == 2:
        batch_size, feat_dim = x.shape
    else:
        # Multi-dimensional input, assume last dimension is features
        batch_size = 1
        for dim in x.shape[:-1]:
            batch_size *= dim
        feat_dim = x.shape[-1]
    
    # Reshape input to 2D for processing
    x_2d = x.reshape(-1, feat_dim)
    
    # Create output tensor
    out = torch.empty_like(x_2d, dtype=torch.float32)
    
    # Block size selection (must match feature dimension)
    BLOCK_SIZE = feat_dim  # Process entire feature row at once
    
    # Number of programs (one per row)
    num_rows = x_2d.shape[0]
    
    if BLOCK_SIZE == feat_dim:
        # Launch kernel with one program per row
        layernorm_kernel[(num_rows,)](
            x_ptr=x_2d,
            gamma_ptr=weight,
            beta_ptr=bias,
            out_ptr=out,
            n_elements=num_rows * feat_dim,
            feat_dim=feat_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Simple fallback: reshape and process row by row without calling layer_norm directly
        # Process each row with the kernel using appropriate block size
        row_idx = 0
        while row_idx < num_rows:
            current_batch = min(BLOCK_SIZE, num_rows - row_idx)
            out_rows = out[row_idx:row_idx + current_batch]
            x_rows = x_2d[row_idx:row_idx + current_batch]
            
            # Small kernel for processing batch of rows
            layernorm_kernel[(current_batch,)](
                x_ptr=x_rows,
                gamma_ptr=weight,
                beta_ptr=bias,
                out_ptr=out_rows,
                n_elements=current_batch * feat_dim,
                feat_dim=feat_dim,
                BLOCK_SIZE=feat_dim,
            )
            row_idx += current_batch
    
    # Reshape output back to original shape
    return out.reshape(x.shape)

def replacement_func():
    return optimized_layernorm