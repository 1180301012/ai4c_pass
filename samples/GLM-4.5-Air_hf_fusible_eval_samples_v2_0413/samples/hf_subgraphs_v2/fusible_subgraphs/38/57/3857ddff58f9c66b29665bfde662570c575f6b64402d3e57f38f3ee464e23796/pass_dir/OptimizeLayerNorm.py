import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match the computation pattern from the model
    # in_0: bias, in_1: weight, in_2, in_3: input tensors
    tmp_2 = in_2 + in_3  # This matches in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, in_1.shape[0])  # This matches reshape(-1, D)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (in_1.shape[0],), in_1, in_0, 1e-05)  # layer_norm with weight, bias, eps
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    n_elements,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    mean = tl.broadcast_to(mean, x)
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_elements
    std = tl.sqrt(var + eps)
    
    # Normalize
    x_normalized = x_centered / std
    
    # Load weight and bias (they are small vectors, so broadcast across iterations)
    weight = tl.load(weight_ptr + (offsets % hidden_size), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + (offsets % hidden_size), mask=mask, other=0.0)
    
    # Apply weight and bias
    out = x_normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)
    
    # Also store the reshaped input for the return value
    tl.store(x_ptr + offsets, x, mask=mask)

@torch.fx.wrap  
def triton_layer_norm(in_0, in_1, in_2, in_3):
    # in_0: bias, in_1: weight, in_2, in_3: input tensors
    # First, do the addition
    tmp_2 = in_2 + in_3
    
    # Then reshape to 2D for layer norm
    hidden_size = in_1.shape[0]
    x_2d = tmp_2.reshape(-1, hidden_size)
    
    N, D = x_2d.shape
    
    # Output is needed for both the rescaled input and layer norm result
    out_layer_norm = torch.empty_like(x_2d, dtype=x_2d.dtype)
    
    BLOCK_SIZE = 1024
    num_programs = (N * D + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    layer_norm_kernel[(num_programs,)](
        x_ptr=x_2d,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=out_layer_norm,
        n_elements=N * D,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x_2d, out_layer_norm

def replacement_func():
    return triton_layer_norm