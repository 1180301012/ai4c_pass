import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match the exact computation from the model
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, in_1.shape[0])
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (in_1.shape[0],), in_1, in_0, 1e-05)
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_layer_norm_kernel(
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
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean using reduction
    mean = tl.sum(x, axis=0) / hidden_size
    mean = tl.broadcast_to(mean, x)
    
    # Compute variance: E[(x - mu)^2]
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    std = tl.sqrt(var + eps)
    
    # Normalize: (x - mu) / sigma
    x_normalized = x_centered / std
    
    # Load weight and bias (small vectors, broadcast across elements)
    weight = tl.load(weight_ptr + (offsets % hidden_size), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + (offsets % hidden_size), mask=mask, other=0.0)
    
    # Apply affine transformation
    out = x_normalized * weight + bias
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(x_ptr + offsets, x, mask=mask)  # Store original reshaped input for return

@torch.fx.wrap
def full_layer_norm_optimization(in_0, in_1, in_2, in_3):
    """
    in_0: bias tensor, in_1: weight tensor, in_2, in_3: input tensors
    """
    # Step 1: Element-wise addition
    tmp_2 = in_2 + in_3
    
    # Step 2: Reshape to 2D for layer norm
    hidden_size = in_1.shape[0]
    x_2d = tmp_2.reshape(-1, hidden_size)
    
    N, D = x_2d.shape
    total_elements = N * D
    
    # Create output tensors
    reshaped_input_for_return = torch.empty_like(x_2d)
    layer_norm_output = torch.empty_like(x_2d)
    
    # Optimized Triton kernel parameters
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    optimized_layer_norm_kernel[(num_programs,)](
        x_ptr=reshaped_input_for_return,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=layer_norm_output,
        n_elements=total_elements,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return reshaped_input_for_return, layer_norm_output

def replacement_func():
    return full_layer_norm_optimization