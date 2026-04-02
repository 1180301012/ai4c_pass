import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    """Pattern: GELU followed by BatchNorm"""
    # GELU activation
    gelu_out = torch.nn.functional.gelu(input_tensor, approximate='none')
    # BatchNorm with exact parameter order from original model
    batchnorm_out = torch.nn.functional.batch_norm(
        gelu_out, running_mean, running_var, weight, bias, 
        training=False, momentum=0.1, eps=1e-05
    )
    return gelu_out, batchnorm_out

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    """Extract arguments for the fused operation"""
    return (input_tensor, running_mean, running_var, weight, bias)

@triton.jit
def fused_gelu_batchnorm_kernel(
    input_ptr, output_ptr, gelu_ptr,
    weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    n_elements, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU + BatchNorm kernel (identity operation is optimized out)"""
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    grid_size = tl.program_id(1)
    
    # Calculate work distribution per program
    work_per_program = (n_elements + grid_size - 1) // grid_size
    start_idx = pid * work_per_program
    end_idx = min((pid + 1) * work_per_program, n_elements)
    
    # Load batch norm parameters (per-channel for spatial dimensions)
    weight = tl.load(weight_ptr + pid)
    bias = tl.load(bias_ptr + pid)
    running_mean = tl.load(running_mean_ptr + pid)
    running_var = tl.load(running_var_ptr + pid)
    
    # Precompute scale and shift for batch norm
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = weight * inv_std
    shift = bias - running_mean * inv_std
    
    # Process data in chunks
    for i in range(start_idx, end_idx):
        # Load input
        x = tl.load(input_ptr + i)
        
        # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = x * x * x
        gelu_x = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x_cubed)))
        
        # Batch normalization: scale * gelu_x + shift
        # Identity operation (0 + batchnorm_x) is optimized out since it's equivalent to batchnorm_x
        batchnorm_x = gelu_x * scale + shift
        
        # Store results (gelu_x and batchnorm_x directly)
        tl.store(gelu_ptr + i, gelu_x)
        tl.store(output_ptr + i, batchnorm_x)

@torch.fx.wrap
def fused_gelu_batchnorm(input_tensor, running_mean, running_var, weight, bias):
    """Wrapper for the fused GELU + BatchNorm operation"""
    N, C, H, W = input_tensor.shape
    n_elements = N * C * H * W
    
    # Create output tensors
    gelu_out = torch.empty_like(input_tensor)
    batchnorm_out = torch.empty_like(input_tensor)
    
    # Get tensor pointers
    input_ptr = input_tensor
    running_mean_ptr = running_mean
    running_var_ptr = running_var
    weight_ptr = weight
    bias_ptr = bias
    
    # Configure kernel launch
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_size = C  # One program per channel for proper parameter access
    
    # Launch kernel
    fused_gelu_batchnorm_kernel[(num_blocks, grid_size)](
        input_ptr, batchnorm_out, gelu_out,
        weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
        n_elements, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return gelu_out, batchnorm_out

def replacement_func():
    """Returns the fused kernel function"""
    return fused_gelu_batchnorm