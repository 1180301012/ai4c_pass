import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    # Linear operation
    tmp_2 = torch.nn.functional.linear(in_3, in_1, in_0)
    # Permute operation
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def linear_permute_kernel(
    x_ptr,         # Input tensor [B, N, K]
    weights_ptr,   # Weights [O, K]
    bias_ptr,      # Bias [O]
    out_ptr,       # Output permute [B*O, N] (flattened)
    B, N, K, O,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Compute total number of output elements
    total_outputs = B * O
    
    # Only process if this program is within valid range
    if pid >= total_outputs:
        return
    
    # Convert flattened program ID to matrix coordinates
    m = pid // O  # Batch index
    n = pid % O   # Output index
    
    # Only process if within valid bounds
    if m >= B or n >= O:
        return
    
    # Initialize accumulator
    acc = tl.zeros((N,), dtype=tl.float32)
    
    # Process matrix multiplication along K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = tl.minimum(k + BLOCK_SIZE_K, K)
        
        # Load weights for current K range
        weights = tl.load(weights_ptr + n * K + tl.arange(k, k_end),
                        mask=tl.arange(k, k_end) < K,
                        other=0.0)
        
        # Load input for current K range
        x = tl.load(x_ptr + m * N * K + n * K + tl.arange(k, k_end),
                   mask=tl.arange(k, k_end) < K,
                   other=0.0)
        
        # Dot product
        acc += x * weights.to(tl.float32)
    
    # Add bias
    bias = tl.load(bias_ptr + n, mask=n < O, other=0.0)
    acc = acc + bias
    
    # Store result
    tl.store(out_ptr + pid, acc)

@torch.fx.wrap
def fused_linear_permute(x, weights, bias):
    B, N, K = x.shape
    O = weights.shape[0]
    
    # Create output tensor - stored in flattened [B*O, N] format
    out = torch.empty((B * O, N), dtype=x.dtype, device=x.device)
    
    # Configure block sizes for optimal GPU utilization
    BLOCK_SIZE_M = 1   # Each program handles one output element
    BLOCK_SIZE_N = 1   # Each program handles one output element  
    BLOCK_SIZE_K = 256  # Larger K tile for better memory access
    
    # Calculate grid size - one program per output element
    grid = B * O
    
    # Launch kernel
    linear_permute_kernel[grid](
        x,
        weights,
        bias,
        out,
        B, N, K, O,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    # Reshape output to [B, O, N] format
    return out.view(B, O, N)

def replacement_func():
    return fused_linear_permute