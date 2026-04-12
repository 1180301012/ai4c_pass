import torch
import triton
import triton.language as tl

def linear_pattern(x, weight, bias):
    """
    Pattern for torch.nn.functional.linear(x, weight, bias)
    x: [N, 384] - input tensor
    weight: [1000, 384] - weight matrix  
    bias: [1000] - bias vector
    Output: [N, 1000]
    """
    result = torch.nn.functional.linear(x, weight, bias)
    return result

def linear_replacement_args(x, weight, bias):
    """
    Extract arguments for linear operation
    """
    return (x, weight, bias)

@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,  # batch size (rows of x)
    N,  # output features (1000)
    K,  # input features (384)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Optimized linear kernel using Triton
    Computes: output = x @ weight.T + bias
    """
    # Get program IDs
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output feature dimension
    
    # Compute offsets within the block
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    m_mask = m_offset < M
    n_mask = n_offset < N
    
    # Load bias vector
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = k + k_offset < K
        
        # Load weight slice and transpose
        weight_slice = tl.load(
            weight_ptr + n_offset[:, None] * K + (k + k_offset)[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Load x slice
        x_slice = tl.load(
            x_ptr + m_offset[:, None] * K + (k + k_offset)[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Matrix multiplication: acc += x_slice @ weight_slice.T
        acc += tl.dot(x_slice, weight_slice, out_layout='MM')
    
    # Add bias and store result
    acc = acc + bias[None, :]
    
    # Store output
    out_ptr_base = out_ptr + m_offset[:, None] * N + n_offset[None, :]
    tl.store(out_ptr_base, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def linear_forward(x, weight, bias):
    """
    Wrapper function to launch the optimized linear kernel
    """
    M, K = x.shape  # M: batch size, K: input features
    N = weight.shape[0]  # N: output features
    
    # Choose block sizes for good GPU occupancy
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Create output tensor
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    linear_kernel[grid](
        x=x,
        weight=weight,
        bias=bias,
        out=out,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def linear_replacement_func():
    """
    Replacement function for linear operation
    """
    return linear_forward