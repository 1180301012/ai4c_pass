import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias):
    """
    Pattern: Layer normalization optimization
    Matches: torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
    """
    normalized_shape = input_tensor.shape[-1]
    result = torch.nn.functional.layer_norm(input_tensor, (normalized_shape,), weight, bias, 1e-05)
    return result

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.jit
def layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate global index
    total_elements = batch_size * seq_len * hidden_dim
    mask = idx < total_elements
    
    # Load input data for this program
    x = tl.load(input_ptr + idx, mask=mask, other=0.0)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + idx % hidden_dim, mask=(idx % hidden_dim) < hidden_dim)
    bias = tl.load(bias_ptr + idx % hidden_dim, mask=(idx % hidden_dim) < hidden_dim)
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / hidden_dim
    tl.store(mean_ptr + idx, mean, mask=mask)
    
    # Calculate rstd (inverse std deviation)
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_dim
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + idx, rstd, mask=mask)
    
    # Apply normalization
    x_norm = x_centered * rstd
    y = x_norm * weight + bias
    
    # Store output
    tl.store(output_ptr + idx, y, mask=mask)

@triton.jit
def layernorm_backward_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate global index
    total_elements = batch_size * seq_len * hidden_dim
    mask = idx < total_elements
    
    # Load data
    grad_output = tl.load(grad_output_ptr + idx, mask=mask, other=0.0)
    x = tl.load(input_ptr + idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + idx % hidden_dim, mask=(idx % hidden_dim) < hidden_dim)
    mean = tl.load(mean_ptr + idx, mask=mask)
    rstd = tl.load(rstd_ptr + idx, mask=mask)
    
    # Backward calculations
    x_centered = x - mean
    norm = x_centered * rstd
    
    # Calculate gradients
    grad_output_weight = grad_output * weight
    grad_norm = grad_output_weight * rstd
    grad_var = tl.sum(grad_norm * x_centered, axis=0) * -0.5 * rstd * rstd * rstd
    grad_mean = tl.sum(grad_norm * -1.0, axis=0) + grad_var * tl.sum(x_centered * 2.0, axis=0) / hidden_dim
    
    grad_input = grad_norm + (grad_var * 2.0 * x_centered + grad_mean) / hidden_dim
    
    # Accumulate weight and bias gradients
    for i in range(hidden_dim):
        mask_weight = (idx % hidden_dim) == i
        tl.atomic_add(grad_weight_ptr + i, tl.sum(grad_output * norm, axis=0), mask=mask_weight)
        tl.atomic_add(grad_bias_ptr + i, tl.sum(grad_output, axis=0), mask=mask_weight)
    
    tl.store(grad_input_ptr + idx, grad_input, mask=mask)

@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias):
    batch_size, seq_len, hidden_dim = input_tensor.shape
    
    # Allocate output and intermediate tensors
    output = torch.empty_like(input_tensor)
    mean = torch.empty_like(input_tensor)
    rstd = torch.empty_like(input_tensor)
    
    # Calculate grid size
    BLOCK_SIZE = 256
    total_elements = batch_size * seq_len * hidden_dim
    grid = ( (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    # Launch kernel
    layernorm_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        mean_ptr=mean,
        rstd_ptr=rstd,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_layernorm