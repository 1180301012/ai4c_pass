import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, eps):
    # Pattern: layer normalization
    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight_tensor, bias_tensor, eps)

def replacement_args(input_tensor, weight_tensor, bias_tensor, eps):
    return (input_tensor, weight_tensor, bias_tensor, eps)

@triton.jit
def layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    n_elements,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel for layer normalization """
    # Each program processes one element per sequence position
    pid = tl.program_id(0)
    
    # Compute offset for this sequence position
    offset = pid * hidden_size
    mask = offset + tl.arange(0, BLOCK_SIZE) < n_elements
    if not mask[0]:
        return  # Only process valid elements
    
    # Load input, gamma, and beta
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x) / hidden_size
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / hidden_size
    
    # Compute inverse standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization, scale, and shift
    y = (x_centered * rstd) * gamma + beta
    
    # Store result
    tl.store(y_ptr + offset, y, mask=mask)

@torch.fx.wrap
def optimized_layernorm(input_tensor, weight_tensor, bias_tensor, eps=1e-12):
    # Get tensor dimensions
    batch_size, seq_len, hidden_size = input_tensor.shape
    n_elements = batch_size * seq_len
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Optimize block size based on hidden dimension
    if hidden_size <= 256:
        BLOCK_SIZE = 64
    elif hidden_size <= 512:
        BLOCK_SIZE = 128
    elif hidden_size <= 1024:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512
    
    # Calculate grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    layernorm_kernel[grid](
        x_ptr=input_tensor,
        gamma_ptr=weight_tensor,
        beta_ptr=bias_tensor,
        y_ptr=output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_layernorm