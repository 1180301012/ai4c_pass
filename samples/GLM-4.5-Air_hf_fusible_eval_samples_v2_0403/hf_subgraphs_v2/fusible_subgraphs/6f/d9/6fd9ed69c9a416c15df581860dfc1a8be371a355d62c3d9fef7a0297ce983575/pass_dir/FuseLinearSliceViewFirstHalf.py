import torch
import triton
import triton.language as tl

def pattern(update_feature, dynamic_weight, dynamic_bias):
    linear = torch.nn.functional.linear(update_feature, dynamic_weight, dynamic_bias)
    tmp_5 = linear[(slice(None, None, None), slice(None, 256, None))]
    tmp_6 = tmp_5.view(-1, 256)
    return tmp_6

def replacement_args(update_feature, dynamic_weight, dynamic_bias):
    return (update_feature, dynamic_weight, dynamic_bias)

@triton.jit
def linear_slice_first_half_kernel(
    update_feature_ptr, dynamic_weight_ptr, dynamic_bias_ptr,
    out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SLICE_SIZE: tl.constexpr,
):
    # Program Ids
    pid_M = tl.program_id(0)
    pid_N = tl.program_id(1)
    
    # Ranges
    m_offsets = pid_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_N * SLICE_SIZE + tl.arange(0, SLICE_SIZE)  # Only process first 256
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    mask_M = m_offsets < M
    mask_N = n_offsets < SLICE_SIZE
    mask_K = k_offsets < K
    
    # Load weight matrix (transposed for gemv)
    w = tl.load(dynamic_weight_ptr + k_offsets[:, None] * N + n_offsets[None, :], mask=mask_K[:, None] & mask_N[None, :], other=0.0)
    
    # Accumulators
    acc = tl.zeros((BLOCK_SIZE_M, SLICE_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        off_k = k + k_offsets
        mask_k = off_k < K
        
        # Load input chunk
        x = tl.load(update_feature_ptr + m_offsets[:, None] * K + off_k[None, :], 
                   mask=mask_M[:, None] & mask_k[None, :], other=0.0)
        
        # Load bias
        bias = tl.load(dynamic_bias_ptr + n_offsets, mask=mask_N, other=0.0)
        bias = bias[None, :]  # Broadcast across M dimension
        
        # Matrix multiplication
        acc += tl.dot(x, w.to(tl.float32), out_dtype=tl.float32)
        acc += bias
    
    # Store result only for the first half (slice(None, 256))
    m_store = m_offsets
    n_store = n_offsets
    
    mask_out = mask_M[:, None] & mask_N[None, :]
    tl.store(out_ptr + m_store[:, None] * SLICE_SIZE + n_store[None, :], acc, mask=mask_out)

@torch.fx.wrap
def fused_linear_slice_first_half(update_feature, dynamic_weight, dynamic_bias):
    # Input shapes: update_feature [M, K], dynamic_weight [K, 256], dynamic_bias [256]
    M, K = update_feature.shape
    SLICE_SIZE = 256
    
    # Output will be [M, SLICE_SIZE] -> view to [M*SLICE_SIZE_factor, 256] but let's compute the right shape
    # Since we're taking the first 256 from the 512 output, and then view(-1, 256), 
    # the result should be [M, 256] which becomes [M*1, 256] when viewed as (-1, 256)
    
    out_shape = (M, SLICE_SIZE)
    out = torch.empty(out_shape, dtype=update_feature.dtype, device=update_feature.device)
    
    # Number of programs
    def cdiv(a, b):
        return (a + b - 1) // b
    
    GRID_M = cdiv(M, 128)  # Use BLOCK_SIZE_M = 128
    GRID_N = cdiv(SLICE_SIZE, 256)  # Each processes all 256 elements of the slice
    
    linear_slice_first_half_kernel[(GRID_M, GRID_N)](
        update_feature,
        dynamic_weight,
        dynamic_bias,
        out,
        M, SLICE_SIZE, K,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=SLICE_SIZE,
        BLOCK_SIZE_K=32,
        SLICE_SIZE=SLICE_SIZE,
    )
    
    # Apply the view operation
    result = out.view(-1, SLICE_SIZE)
    return result

def replacement_func():
    return fused_linear_slice_first_half