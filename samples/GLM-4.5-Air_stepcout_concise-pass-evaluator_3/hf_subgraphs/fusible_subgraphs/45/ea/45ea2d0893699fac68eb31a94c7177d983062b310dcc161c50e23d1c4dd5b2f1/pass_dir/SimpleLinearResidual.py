import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    # Simple pattern: linear + dropout + addition
    linear_output = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    dropout_output = torch.nn.functional.dropout(linear_output, p=0.0, training=False)
    final_output = residual_tensor + dropout_output
    return final_output, dropout_output

def replacement_args(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    return (input_tensor, weight_tensor, bias_tensor, residual_tensor)

@triton.jit
def simple_linear_kernel(
    x_ptr, w_ptr, b_ptr, residual_ptr, out_ptr,
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
    
    # Add bias and residual
    acc = acc + b[None, :] + tl.load(residual_ptr + m_offsets[:, None] * K + k_offsets[None, :], 
                                   mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    # Store result
    tl.store(out_ptr + m_offsets[:, None] * K + k_offsets[None, :], 
             acc, mask=m_mask[:, None] & k_mask[None, :])

@torch.fx.wrap
def simple_linear_optimized(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    M, N = input_tensor.shape
    K = weight_tensor.shape[0]
    
    # Handle device placement
    devices = [input_tensor, weight_tensor, bias_tensor, residual_tensor]
    if any(t.device.type != 'cuda' for t in devices):
        devices = [t.cuda() if t.device.type != 'cuda' else t for t in devices]
        input_tensor, weight_tensor, bias_tensor, residual_tensor = devices
    
    output = torch.empty((M, K), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block sizes for different scenarios
    if N == 128 and M >= 1000:  # LINKX
        BLOCK_SIZE_M, BLOCK_SIZE_K = 64, 32
    else:  # BERT-like
        BLOCK_SIZE_M, BLOCK_SIZE_K = 32, 32
    
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    simple_linear_kernel[(grid_m, grid_k)](
        input_tensor, weight_tensor, bias_tensor, residual_tensor, output,
        M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    
    return output, output

def replacement_func():
    return simple_linear_optimized