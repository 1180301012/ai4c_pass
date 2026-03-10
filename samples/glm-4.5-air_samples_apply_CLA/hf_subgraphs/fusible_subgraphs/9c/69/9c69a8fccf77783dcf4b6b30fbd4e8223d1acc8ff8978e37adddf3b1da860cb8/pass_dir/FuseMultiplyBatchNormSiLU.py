import torch
import triton
import triton.language as tl

def pattern(x, w):
    # Simple element-wise multiplication
    return x * w

def replacement_args(x, w):
    return (x, w)

@triton.jit
def fused_multiply_bnorm_silu_kernel(
    # Input pointers
    x_ptr,
    sigmoid_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    # Output pointers
    multiplied_out_ptr,
    bn_out_ptr,
    silu_out_ptr,
    # Tensor metadata
    N, C, H, W,
    eps: tl.constexpr = 1e-05,
    momentum: tl.constexpr = 0.1,
    BLOCK_SIZE_C: tl.constexpr = 32,
    BLOCK_SIZE_HW: tl.constexpr = 16
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Handle channels
    c_start = pid_c * BLOCK_SIZE_C
    c_end = min((pid_c + 1) * BLOCK_SIZE_C, C)
    
    # Create pointers for this block
    x_ptr_block = x_ptr + pid_h * W + pid_w * (C * H * W)
    sigmoid_weight_ptr_block = sigmoid_weight_ptr + pid_w * C
    
    # Load batch norm parameters for this channel block
    mean_ptr_block = running_mean_ptr + c_start
    var_ptr_block = running_var_ptr + c_start
    weight_ptr_block = weight_ptr + c_start
    bias_ptr_block = bias_ptr + c_start
    
    # Load parameters (coalesced memory access)
    running_mean = tl.load(mean_ptr_block)
    running_var = tl.load(var_ptr_block)
    gamma = tl.load(weight_ptr_block)
    beta = tl.load(bias_ptr_block)
    
    # Compute variance inverse (1/stddev)
    # Add epsilon for numerical stability
    inv_var = tl.reciprocal(tl.sqrt(running_var + eps))
    
    # Load sigmoid weight (broadcast across spatial dimensions)
    sigmoid_weight = tl.load(sigmoid_weight_ptr_block)
    
    # Process each channel block
    for c in range(c_start, c_end):
        # Load input element
        x = tl.load(x_ptr_block + c, mask=c < C, other=0.0)
        
        # Step 1: Element-wise multiplication
        multiplied = x * sigmoid_weight
        
        # Store intermediate result if needed (optional)
        tl.store(multiplied_out_ptr + pid_h * W + pid_w * (C * H * W) + c, multiplied, mask=c < C)
        
        # Step 2: Batch normalization
        # (x - mean) * (gamma / sqrt(var + eps)) + beta
        normalized = (multiplied - running_mean) * gamma * inv_var + beta
        
        # Store batch norm output
        tl.store(bn_out_ptr + pid_h * W + pid_w * (C * H * W) + c, normalized, mask=c < C)
        
        # Step 3: SiLU activation (x * sigmoid(x))
        silu_out = normalized * tl.sigmoid(normalized)
        
        # Store SiLU output
        tl.store(silu_out_ptr + pid_h * W + pid_w * (C * H * W) + c, silu_out, mask=c < C)

@triton.jit
def simple_multiply_kernel(x_ptr, w_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data 
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data and compute multiplication
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    out = x * w
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_multiply(x, w):
    # Handle broadcasting case
    if w.dim() == 0:
        return x * w
    
    # For non-scalar w, create a simple Triton kernel
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_multiply_kernel[(num_programs,)](
        x, w, out, N, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return simple_multiply