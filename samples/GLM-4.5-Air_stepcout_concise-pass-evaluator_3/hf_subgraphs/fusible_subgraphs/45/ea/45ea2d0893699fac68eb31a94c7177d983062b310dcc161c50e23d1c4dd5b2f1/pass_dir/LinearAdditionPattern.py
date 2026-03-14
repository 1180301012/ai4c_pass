import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    # Simple linear + addition pattern matching the core computation
    def pattern_implementation(a, b, c, d):
        tmp_0 = a
        tmp_1 = b
        tmp_2 = c
        tmp_3 = torch.nn.functional.linear(tmp_2, tmp_1, tmp_0)
        tmp_5 = d + tmp_3
        return (tmp_5, tmp_3)
    
    return pattern_implementation(input_tensor, weight_tensor, bias_tensor, residual_tensor)

def replacement_args(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    return (input_tensor, weight_tensor, bias_tensor, residual_tensor)

@triton.jit
def linear_addition_kernel(
    x_ptr, w_ptr, b_ptr, residual_ptr, out_ptr, intermediate_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    m_mask = m_offsets < M
    k_mask = k_offsets < K
    
    # Load weights and bias
    w = tl.load(w_ptr + k_offsets[:, None] * N, mask=k_mask[:, None], other=0.0)
    b = tl.load(b_ptr + k_offsets, mask=k_mask, other=0.0)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Matrix multiplication
    for n in range(0, N):
        x = tl.load(x_ptr + m_offsets * N + n, mask=m_mask, other=0.0)
        acc += x[:, None] * w
    
    # Add bias
    acc = acc + b[None, :]
    
    # Store intermediate result (linear output)
    tl.store(intermediate_ptr + m_offsets[:, None] * K + k_offsets[None, :], 
             acc, mask=m_mask[:, None] & k_mask[None, :])
    
    # Add residual and store final result
    result = acc + tl.load(residual_ptr + m_offsets[:, None] * K + k_offsets[None, :], 
                          mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    tl.store(out_ptr + m_offsets[:, None] * K + k_offsets[None, :], 
             result, mask=m_mask[:, None] & k_mask[None, :])

@torch.fx.wrap
def linear_addition_optimized(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    M, N = input_tensor.shape
    K = weight_tensor.shape[0]
    
    # Handle device placement
    devices = [input_tensor, weight_tensor, bias_tensor, residual_tensor]
    if any(t.device.type != 'cuda' for t in devices):
        devices = [t.cuda() if t.device.type != 'cuda' else t for t in devices]
        input_tensor, weight_tensor, bias_tensor, residual_tensor = devices
    
    output = torch.empty((M, K), dtype=input_tensor.dtype, device=input_tensor.device)
    intermediate = torch.empty((M, K), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Auto-tune block sizes based on tensor sizes
    if N <= 4 and K <= 4:  # Very small BERT-like
        BLOCK_SIZE_M, BLOCK_SIZE_K = 8, 4
    elif N == 128 and M >= 1000:  # LINKX-like
        BLOCK_SIZE_M, BLOCK_SIZE_K = 64, 32
    elif N == 512 and M == 4:  # Medium BERT-like
        BLOCK_SIZE_M, BLOCK_SIZE_K = 4, 64
    else:  # Default
        BLOCK_SIZE_M, BLOCK_SIZE_K = 32, 32
    
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    linear_addition_kernel[(grid_m, grid_k)](
        input_tensor, weight_tensor, bias_tensor, residual_tensor, output, intermediate,
        M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    
    # Return both final output and intermediate (to match pattern return)
    return output, intermediate

def replacement_func():
    return linear_addition_optimized