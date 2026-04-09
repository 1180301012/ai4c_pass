import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: Linear + Add + ReLU
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Highly optimized kernel using efficient threading
@triton.jit
def fused_linear_add_relu_kernel(
    x_ptr,           # in_3: input [M, K]
    w_ptr,           # in_1: weights [K, N]  
    b_ptr,           # in_0: bias [N]
    y_ptr,           # in_2: skip connection [M, N]
    out_ptr,         # output [M, N]
    M, K, N,         # dimensions
):
    # Each thread handles one output element - optimal for our problem size
    pid = tl.program_id(0)
    
    # Check bounds
    if pid >= M * N:
        return
    
    # Convert linear index to 2D coordinates
    m_idx = pid // N
    n_idx = pid % N
    
    # Optimized dot product computation
    acc = 0.0
    
    # Vectorized computation along K dimension
    for k in range(K):
        # Load input element - row-major order for good cache locality
        x_offset = m_idx * K + k
        x_val = tl.load(x_ptr + x_offset, mask=True, other=0.0).to(tl.float32)
        
        # Load weight element - column access, use stride pattern
        w_offset = k * N + n_idx
        w_val = tl.load(w_ptr + w_offset, mask=True, other=0.0).to(tl.float32)
        
        # Accumulate with FMA (fused multiply-add)
        acc = acc + x_val * w_val
    
    # Load bias - single value per column
    b_val = tl.load(b_ptr + n_idx, mask=True, other=0.0).to(tl.float32)
    
    # Load skip connection
    y_offset = m_idx * N + n_idx
    y_val = tl.load(y_ptr + y_offset, mask=True, other=0.0).to(tl.float32)
    
    # Final computation with ReLU
    result = acc + b_val + y_val
    result = tl.maximum(result, 0.0)
    
    # Convert to original precision efficiently
    result = result.to(tl.float16 if x_ptr.dtype.element_ty == tl.float16 else tl.bfloat16)
    
    # Store directly
    tl.store(out_ptr + pid, result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_linear_add_relu(in_0, in_1, in_2, in_3):
    """Fused Linear + Add + ReLU operation"""
    M, K = in_3.shape  # in_3: [M, K]
    N = in_1.shape[1]  # in_1: [K, N] -> output dim is N
    
    # Calculate grid size - one thread per output element
    total_elements = M * N
    
    # Create output tensor
    out = torch.empty((M, N), dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel with one thread per output element
    fused_linear_add_relu_kernel[total_elements,](
        in_3, in_1, in_0, in_2, out,
        M, K, N
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_add_relu