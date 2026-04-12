import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match ReLU followed by BatchNorm pattern
    Original computation:
    tmp_4 = torch.nn.functional.relu(in_4, inplace = False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    """
    tmp_4 = torch.nn.functional.relu(in_4, inplace = False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for the replacement function
    We need all the inputs and a route identifier
    """
    return (in_0, in_1, in_2, in_3, in_4, "relu_batchnorm_fusion")

# Triton kernel for fused ReLU + BatchNorm
@triton.jit
def fused_relu_batchnorm_kernel(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr, 
    output_ptr,
    num_features,
    num_elements,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load normalization parameters
    running_mean = tl.load(running_mean_ptr + tl.arange(0, num_features)[:, None], mask=(tl.arange(0, num_features)[:, None] < num_features), other=0.0)
    running_var = tl.load(running_var_ptr + tl.arange(0, num_features)[:, None], mask=(tl.arange(0, num_features)[:, None] < num_features), other=1.0)
    weight = tl.load(weight_ptr + tl.arange(0, num_features)[:, None], mask=(tl.arange(0, num_features)[:, None] < num_features), other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, num_features)[:, None], mask=(tl.arange(0, num_features)[:, None] < num_features), other=0.0)
    
    # Reshape parameters for broadcasting
    running_mean = tl.broadcast_to(running_mean, (num_features, BLOCK_SIZE))
    running_var = tl.broadcast_to(running_var, (num_features, BLOCK_SIZE))
    weight = tl.broadcast_to(weight, (num_features, BLOCK_SIZE))
    bias = tl.broadcast_to(bias, (num_features, BLOCK_SIZE))
    
    # Apply normalization
    x_normalized = (x - running_mean) / tl.sqrt(running_var + eps)
    
    # Apply weight and bias
    x_scaled = x_normalized * weight + bias
    
    # Apply ReLU
    x_relu = tl.maximum(x_scaled, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, x_relu, mask=mask)

@torch.fx.wrap
def fused_relu_batchnorm(running_mean, running_var, weight, bias, input_tensor):
    """
    Fused ReLU + BatchNorm operation using Triton kernel
    """
    num_features = running_mean.shape[0]
    num_elements = input_tensor.numel()
    BLOCK_SIZE = 1024  # Can be tuned
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    fused_relu_batchnorm_kernel[(num_programs,)](
        input_tensor,
        running_mean,
        running_var, 
        weight,
        bias,
        output,
        num_features,
        num_elements,
        1e-05,  # eps
        0.1,    # momentum (not used in inference, but kept for consistency)
        BLOCK_SIZE
    )
    
    return output

@torch.fx.wrap
def dispatch_wrapper(*args):
    """
    Dispatch wrapper that routes to the appropriate optimization based on route identifier
    """
    # Last argument is the route identifier
    route = args[-1]
    if route == "dropout_elimination":
        return args[0]  # Just return input for dropout elimination
    elif route == "relu_batchnorm_fusion":
        # Route to fused ReLU + BatchNorm
        return fused_relu_batchnorm(args[0], args[1], args[2], args[3], args[4])
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    """
    Returns the optimized kernel wrapper function
    Must be identical across all passes due to replacement_func_limit constraint
    """
    return dispatch_wrapper