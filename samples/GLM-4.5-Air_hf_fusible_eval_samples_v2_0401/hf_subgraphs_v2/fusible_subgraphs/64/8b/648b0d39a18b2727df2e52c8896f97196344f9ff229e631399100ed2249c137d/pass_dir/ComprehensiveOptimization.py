import torch
import triton
import triton.language as tl

def pattern(x, y, running_mean, running_var, weight, bias):
    """Pattern: Input addition followed by GELU and BatchNorm"""
    # Input addition
    added = x + y
    # GELU activation
    gelu_out = torch.nn.functional.gelu(added, approximate='none')
    # BatchNorm
    batchnorm_out = torch.nn.functional.batch_norm(
        gelu_out, running_mean, running_var, weight, bias, 
        training=False, momentum=0.1, eps=1e-05
    )
    return added, gelu_out, batchnorm_out

def replacement_args(x, y, running_mean, running_var, weight, bias):
    """Extract arguments for the fused operation"""
    return (x, y, running_mean, running_var, weight, bias)

@triton.jit
def comprehensive_opt_kernel(
    x_ptr, y_ptr, added_ptr, gelu_ptr, batchnorm_ptr,
    weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    n_elements, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Comprehensive kernel: Input addition + GELU + BatchNorm"""
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    grid_size = tl.program_id(1)
    
    # Calculate work distribution per program
    work_per_program = (n_elements + grid_size - 1) // grid_size
    start_idx = pid * work_per_program
    end_idx = min((pid + 1) * work_per_program, n_elements)
    
    # Load batch norm parameters (per-channel)
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
        # Load inputs
        x_val = tl.load(x_ptr + i)
        y_val = tl.load(y_ptr + i)
        
        # Input addition
        added_val = x_val + y_val
        
        # GELU activation
        x_cubed = added_val * added_val * added_val
        gelu_val = 0.5 * added_val * (1.0 + tl.tanh(0.7978845608028654 * (added_val + 0.044715 * x_cubed)))
        
        # Batch normalization
        batchnorm_val = gelu_val * scale + shift
        
        # Store results
        tl.store(added_ptr + i, added_val)
        tl.store(gelu_ptr + i, gelu_val)
        tl.store(batchnorm_ptr + i, batchnorm_val)

@torch.fx.wrap
def comprehensive_opt(x, y, running_mean, running_var, weight, bias):
    """Wrapper for comprehensive optimization"""
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    
    # Create output tensors
    added_out = torch.empty_like(x)
    gelu_out = torch.empty_like(x)
    batchnorm_out = torch.empty_like(x)
    
    # Get tensor pointers
    x_ptr = x
    y_ptr = y
    weight_ptr = weight
    bias_ptr = bias
    running_mean_ptr = running_mean
    running_var_ptr = running_var
    
    # Configure kernel launch
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_size = C  # One program per channel
    
    # Launch kernel
    comprehensive_opt_kernel[(num_blocks, grid_size)](
        x_ptr, y_ptr, added_out, gelu_out, batchnorm_out,
        weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
        n_elements, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return added_out, gelu_out, batchnorm_out

def replacement_func():
    """Returns the comprehensive optimization function"""
    return comprehensive_opt