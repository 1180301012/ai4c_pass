import torch
import triton
import triton.language as tl

# Pattern matching function - matches layer_norm followed by dropout
def pattern(in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12):
    # Use input variables as they are passed
    _ = in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12
    
    # Layer norm and dropout operations
    result = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    result = torch.nn.functional.dropout(result, p=0.1, training=False)
    
    return result

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12):
    return (tmp_10, in_3, in_2, 1e-05, 0.1)

# Triton kernel for fused layer norm + dropout
@triton.jit
def fused_layernorm_dropout_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    n_elements,
    embed_dim,
    eps,
    dropout_scale,  # (1 - dropout_prob) for inference mode dropout
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / n_elements
    
    # Calculate variance
    x_centered = x - mean
    x2 = x_centered * x_centered
    var = tl.sum(x2, axis=0) / n_elements + eps
    
    # Layer normalization
    x_norm = x_centered / tl.sqrt(var)
    
    # Load gamma and beta parameters
    gamma = tl.load(gamma_ptr + tl.arange(0, embed_dim), mask=tl.arange(0, embed_dim) < embed_dim, other=1.0)
    beta = tl.load(beta_ptr + tl.arange(0, embed_dim), mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
    
    # Apply scaling and shifting
    x_normalized = x_norm * gamma + beta
    
    # Apply dropout scaling (since training=False, this is just multiplication by (1-p))
    out = x_normalized * dropout_scale
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_layernorm_dropout(input_tensor, weight, bias):
    # Get input tensor properties
    n_elements = input_tensor.numel()
    embed_dim = input_tensor.shape[-1]  # Last dimension is 256
    
    # Determine output dtype based on input (should match bfloat16/float16)
    output_dtype = input_tensor.dtype
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Set up Triton kernel launch
    BLOCK_SIZE = 256  # Should be divisible by embed_dim for good performance
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_layernorm_dropout_kernel[(num_programs,)](
        x_ptr=input_tensor,
        gamma_ptr=weight,
        beta_ptr=bias,
        out_ptr=output,
        n_elements=n_elements,
        embed_dim=embed_dim,
        eps=1e-05,
        dropout_scale=0.9,  # 1 - 0.1 for p=0.1 dropout in inference mode
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_layernorm_dropout