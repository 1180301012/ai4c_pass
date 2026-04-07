import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Pattern: dtype conversion + linear - exact match from RECT_L pattern after dropout elimination
    converted = x.to(torch.float16)  # Explicit conversion to float16
    linear = torch.nn.functional.linear(converted, weight, bias)
    return linear

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def fused_conversion_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Convert to the target dtype implicitly through loading
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = m_range < M
    mask_n = n_range < N
    
    # Load bias once (broadcasted across K)
    bias_val = tl.load(bias_ptr + n_range, mask=mask_n)
    
    # Compute result for each block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute k range with masking
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_range < K
        
        # Load x row (converted implicitly through kernel dtype)
        x_ptr_local = x_ptr + m_range[:, None] * K + k_range[None, :]
        x_val = tl.load(x_ptr_local, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load weight columns
        weight_ptr_local = weight_ptr + k_range[:, None] * N + n_range[None, :]
        weight_val = tl.load(weight_ptr_local, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(x_val, weight_val.to(tl.float32))
    
    # Add bias and convert back to original output type
    out = acc + bias_val[None, :]
    final_out = out.to(tl.float16 if weight.dtype == torch.float16 else tl.bfloat16)
    
    # Store results
    out_ptr_local = out_ptr + m_range[:, None] * N + n_range[None, :]
    tl.store(out_ptr_local, final_out, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def fused_conversion_linear(x, weight, bias):
    M, N, K = x.shape[0], weight.shape[1], weight.shape[0]
    
    # Use optimal block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Output type matches weight dtype
    output_dtype = weight.dtype
    
    # Create output tensor
    out = torch.empty((M, N), dtype=output_dtype, device=x.device)
    
    # Launch kernel
    fused_conversion_linear_kernel[(grid_m, grid_n)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        M=M, N=N, K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return fused_conversion_linear