import torch
import triton
import triton.language as tl

# Pattern 1: Independent linear + element-wise operations (mmpose/rtmpose-l pattern)
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)  
    tmp_3 = in_2 * in_1
    return linear, tmp_3

# Argument extraction for Pattern 1
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized Triton kernel for element-wise multiplication
@triton.jit
def elementwise_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x * y
    
    tl.store(out_ptr + offsets, out, mask=mask)

# Optimized Triton kernel for linear transformation
@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offset = (pid % ((batch_size * seq_len + BLOCK_M - 1) // BLOCK_M)) * BLOCK_M
    
    m_indices = m_offset + tl.arange(0, BLOCK_M)
    n_indices = tl.arange(0, BLOCK_N)
    k_indices = tl.arange(0, BLOCK_K)
    
    mask_m = m_indices < (batch_size * seq_len)
    mask_n = n_indices < out_features
    
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_K):
        k_mask = k + k_indices < in_features
        
        x_ptrs = x_ptr + (m_indices.unsqueeze(1) * in_features + k + k_indices)
        weight_ptrs = weight_ptr + (n_indices.unsqueeze(1) * in_features + k + k_indices)
        
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
        
        accum += tl.dot(x, weight.to(tl.float32), out_features=BLOCK_K)
    
    accum = accum.to(tl.float16)
    
    out_ptrs = out_ptr + (m_indices.unsqueeze(1) * out_features + n_indices)
    tl.store(out_ptrs, accum, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def optimized_linear_and_mul(in_0, in_1, in_2, in_3):
    # Element-wise multiplication: in_2 * in_1
    elements = in_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    mul_out = torch.empty_like(in_2)
    elementwise_mul_kernel[(num_programs,)](
        in_2, in_1, mul_out, elements, BLOCK_SIZE
    )
    
    # Linear transformation: torch.nn.functional.linear(in_3, in_0, None)
    if in_3.dim() == 3:
        batch_size, seq_len, in_features = in_3.shape
    else:
        batch_size = 1
        seq_len = in_3.shape[0] if len(in_3.shape) > 1 else 1
        in_features = in_3.shape[-1]
    
    out_features = in_0.shape[0]  # in_0 is [out_features, in_features]
    
    linear_out = torch.empty(batch_size, seq_len, out_features, device=in_3.device, dtype=in_3.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 256
    BLOCK_K = 32
    
    grid_m = (batch_size * seq_len + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_features + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m * grid_n,)
    
    linear_kernel[grid](
        in_3, in_0, linear_out,
        batch_size, seq_len, in_features, out_features,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return (mul_out, linear_out)

def replacement_func():
    return optimized_linear_and_mul