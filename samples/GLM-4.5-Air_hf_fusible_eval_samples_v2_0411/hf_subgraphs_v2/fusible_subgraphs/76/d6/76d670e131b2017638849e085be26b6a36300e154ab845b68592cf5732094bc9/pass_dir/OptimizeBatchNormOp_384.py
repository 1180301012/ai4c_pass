import torch
import triton
import triton.language as tl

def batch_norm_pattern(input, running_mean, running_var, weight, bias):
    """
    Pattern for torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training=False, eps=1e-05)
    input: [N, 384] - input tensor
    running_mean: [384] - mean vector
    running_var: [384] - variance vector  
    weight: [384] - scale vector
    bias: [384] - shift vector
    Output: [N, 384]
    """
    result = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return result

def batch_norm_replacement_args(input, running_mean, running_var, weight, bias):
    """
    Extract arguments for batch norm operation
    """
    return (input, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,  # batch size
    C,  # features (384)
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    """
    Optimized batch norm kernel using Triton
    Computes: output = normalize(input, mean, var, eps) * weight + bias
    """
    # Get program IDs
    pid_n = tl.program_id(0)  # batch dimension
    pid_c = tl.program_id(1)  # feature dimension
    
    # Compute offsets within the block
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offset = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    
    # Create masks
    n_mask = n_offset < N
    c_mask = c_offset < C
    
    # Load pre-computed statistics and parameters
    mean = tl.load(mean_ptr + c_offset, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offset, mask=c_mask, other=0.0).to(tl.float32)
    weight_val = tl.load(weight_ptr + c_offset, mask=c_mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + c_offset, mask=c_mask, other=0.0).to(tl.float32)
    
    # Pre-compute normalization factors
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = weight_val * inv_std
    bias_shift = bias_val - mean * scale
    
    # Process input data in blocks
    for block_start in range(0, C, BLOCK_SIZE_C):
        # Load input slice
        input_slice = tl.load(
            input_ptr + n_offset[:, None] * C + (block_start + c_offset)[None, :],
            mask=n_mask[:, None] & c_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Apply normalization: (x - mean) * scale + bias_shift
        # For better performance, compute bias_shift outside kernel
        normalized = (input_slice - mean[None, :]) * scale[None, :] + bias_shift[None, :]
        
        # Store result for this block
        out_ptr_base = out_ptr + n_offset[:, None] * C + (block_start + c_offset)[None, :]
        tl.store(out_ptr_base, normalized.to(tl.float16), mask=n_mask[:, None] & c_mask[None, :])

@torch.fx.wrap
def batch_norm_forward(input, running_mean, running_var, weight, bias):
    """
    Wrapper function to launch the optimized batch norm kernel
    """
    N, C = input.shape  # N: batch size, C: channels/features (384)
    eps = 1e-05
    
    # Choose block sizes for good GPU occupancy
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_C = 32
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid = (grid_n, grid_c)
    
    # Create output tensor
    out = torch.empty((N, C), dtype=input.dtype, device=input.device)
    
    # Launch kernel
    batch_norm_kernel[grid](
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        out=out,
        N=N,
        C=C,
        eps=eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return out

def batch_norm_replacement_func():
    """
    Replacement function for batch norm operation
    """
    return batch_norm_forward