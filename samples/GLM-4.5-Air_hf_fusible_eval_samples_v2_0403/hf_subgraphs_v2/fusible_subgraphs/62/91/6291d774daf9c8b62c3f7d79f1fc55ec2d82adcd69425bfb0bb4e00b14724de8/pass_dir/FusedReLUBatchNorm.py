import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, momentum=0.1, eps=1e-05):
    """
    Pattern to match ReLU followed by BatchNorm operations
    """
    # Apply ReLU first (not inplace to match original behavior)
    relu_out = torch.nn.functional.relu(x, inplace=False)
    # Apply BatchNorm on the ReLU output
    batch_norm_out = torch.nn.functional.batch_norm(relu_out, running_mean, running_var, weight, bias, momentum, eps)
    return batch_norm_out

def replacement_args(x, running_mean, running_var, weight, bias, momentum=0.1, eps=1e-05):
    return (x, running_mean, running_var, weight, bias, momentum, eps)

@triton.jit
def fused_relu_batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    feature_dim,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch
    pid = tl.program_id(0)
    
    # Calculate start and end indices for this program
    start_idx = pid * BLOCK_SIZE
    end_idx = min((pid + 1) * BLOCK_SIZE, n_elements)
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters once per program (they're the same for all elements in the feature dimension)
    if running_mean_ptr is not None:
        running_mean = tl.load(running_mean_ptr)
    else:
        running_mean = 0.0
    
    if running_var_ptr is not None:
        running_var = tl.load(running_var_ptr)
    else:
        running_var = 1.0
    
    if weight_ptr is not None:
        weight = tl.load(weight_ptr)
    else:
        weight = 1.0
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr)
    else:
        bias = 0.0
    
    # Apply fused ReLU + BatchNorm
    # y = relu((x - running_mean) / sqrt(running_var + eps)) * weight + bias
    
    # Numerically stable variance calculation
    running_var = tl.maximum(running_var, eps)
    
    # Normalization
    x_normalized = (x - running_mean) / tl.sqrt(running_var)
    
    # ReLU
    x_relu = tl.maximum(x_normalized, 0.0)
    
    # Scale and shift
    y = x_relu * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def fused_relu_batch_norm(x, running_mean, running_var, weight, bias, momentum=0.1, eps=1e-05):
    """
    Optimized fused ReLU + BatchNorm operation
    """
    # Handle parameter tensors - move to GPU if needed
    device = x.device
    
    # Ensure parameters are on the same device as input
    if running_mean.device != device:
        running_mean = running_mean.to(device)
    if running_var.device != device:
        running_var = running_var.to(device)
    if weight.device != device:
        weight = weight.to(device)
    if bias.device != device:
        bias = bias.to(device)
    
    # Get tensor dimensions
    batch_dim, feature_dim = x.shape
    n_elements = batch_dim * feature_dim
    
    # Choose block size based on tensor size
    if n_elements <= 1024:
        BLOCK_SIZE = n_elements
        num_programs = 1
    elif n_elements <= 65536:
        BLOCK_SIZE = 256
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Handle None parameters (edge case)
    running_mean_ptr = running_mean.data_ptr() if running_mean is not None else None
    running_var_ptr = running_var.data_ptr() if running_var is not None else None
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch the kernel
    fused_relu_batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean_ptr,
        running_var_ptr=running_var_ptr,
        weight_ptr=weight_ptr,
        bias_ptr=bias_ptr,
        out_ptr=out,
        n_elements=n_elements,
        feature_dim=feature_dim,
        momentum=momentum,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_batch_norm