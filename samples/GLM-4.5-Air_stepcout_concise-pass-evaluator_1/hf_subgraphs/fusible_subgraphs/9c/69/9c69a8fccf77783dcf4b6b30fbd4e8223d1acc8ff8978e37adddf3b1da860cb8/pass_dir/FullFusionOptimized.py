import torch
import triton
import triton.language as tl

# Pattern matching function for the complete fusion: multiply + batch_norm + silu
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern matches the full computation: multiply + batch_norm + silu
    """
    t0 = in_0
    t1 = in_1
    t2 = in_2
    t3 = in_3
    t4 = in_5 * in_4
    t5 = torch.nn.functional.batch_norm(t4, t0, t1, t3, t2, False, 0.1, 1e-05)
    t6 = torch.nn.functional.silu(t5, inplace=True)
    return (t6,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Fully fused Triton kernel for maximum performance
@triton.jit
def full_fusion_kernel(
    # Input tensors
    scale_ptr,      # Scale tensor
    x_ptr,          # Main input tensor
    running_mean_ptr, # running mean
    running_var_ptr,  # running var  
    weight_ptr,     # weight
    bias_ptr,       # bias
    out_ptr,
    # Shape info
    n_elements,
    eps: tl.constexpr,
    
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all inputs
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # For batch norm, we need scalar parameters (broadcast to all elements)
    mean = tl.load(running_mean_ptr + 0, other=0.0).to(tl.float32)
    var = tl.load(running_var_ptr + 0, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + 0, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + 0, other=0.0).to(tl.float32)
    
    # Step 1: Batch norm
    normalized = (x - mean) * tl.math.rsqrt(var + eps) * weight + bias
    
    # Step 2: Element-wise multiplication
    mul_result = normalized * scale
    
    # Step 3: SiLU activation (optimized: x * sigmoid(x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-mul_result))
    silu_result = mul_result * sigmoid_x
    
    # Store final result
    tl.store(out_ptr + offsets, silu_result, mask=mask)

@torch.fx.wrap
def full_fusion_triton(in_0, in_1, in_2, in_3, in_4, in_5):
    n_elements = in_5.numel()
    
    # Optimal block size for full fusion workload
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_5)
    
    full_fusion_kernel[(grid_size,)](
        in_4, in_5, in_0, in_1, in_3, in_2,
        out, n_elements, 1e-05,
        BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return full_fusion_triton