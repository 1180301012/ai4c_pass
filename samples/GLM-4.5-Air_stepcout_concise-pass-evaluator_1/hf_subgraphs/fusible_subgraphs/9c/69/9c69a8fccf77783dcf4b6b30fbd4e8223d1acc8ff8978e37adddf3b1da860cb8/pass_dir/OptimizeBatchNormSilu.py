import torch
import triton
import triton.language as tl

# Pattern matching function for batch_norm + silu
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern matches: batch_norm + silu activation
    """
    t4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    t6 = torch.nn.functional.silu(t4, inplace=True)
    return (t6,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

# Optimized Triton kernel for batch norm + SiLU fusion
@triton.jit
def batch_norm_silu_kernel(
    # Input tensors
    x_ptr,          # Input tensor
    running_mean_ptr, # running mean
    running_var_ptr,  # running var
    weight_ptr,     # weight
    bias_ptr,       # bias
    out_ptr,
    # Shape info
    n_elements,
    # Parameters
    eps: tl.constexpr,
    
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load batch norm parameters
    mean = tl.load(running_mean_ptr + 0, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + 0, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + 0, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + 0, other=0.0).to(tl.float32) 
    
    # Batch norm computation
    normalized = (x - mean) * tl.math.rsqrt(var + eps) * weight + bias
    
    # SiLU activation: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-normalized))
    silu_result = normalized * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, silu_result, mask=mask)

@torch.fx.wrap
def batch_norm_silu_triton(in_0, in_1, in_2, in_3, in_4):
    n_elements = in_4.numel()
    
    # Optimal block size for this workload
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_4)
    
    batch_norm_silu_kernel[(grid_size,)](
        in_4, in_0, in_1, in_3, in_2,
        out, n_elements, 1e-05,
        BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return batch_norm_silu_triton