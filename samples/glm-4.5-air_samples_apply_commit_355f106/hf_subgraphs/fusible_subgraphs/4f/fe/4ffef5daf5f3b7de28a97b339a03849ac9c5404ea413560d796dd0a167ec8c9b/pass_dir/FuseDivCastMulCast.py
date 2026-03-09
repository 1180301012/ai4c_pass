import torch
import triton
import triton.language as tl


# LayerNorm kernel - computes layer norm along the last dimension (K=320)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_div_cast_mul_layernorm_kernel(
    in_4_ptr, in_3_ptr, in_0_ptr, in_2_ptr, in_1_ptr, out_tmp7_ptr, out_tmp8_ptr,
    M, N, K,
    stride_in4_m, stride_in4_n, stride_in4_k,
    stride_in3_m,
    stride_in0_m, stride_in0_n,
    stride_in2_k,
    stride_in1_k,
    stride_out_m, stride_out_n, stride_out_k,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel with LayerNorm: computes (in_4 / in_3).to(float32) * in_0.unsqueeze(-1).to(float32) then layer_norm"""
    pid = tl.program_id(0)
    
    # Compute the (m, n) position this program handles
    # Each program handles one position along M*N, computing the full K output
    mn_idx = pid
    m_idx = mn_idx // N
    n_idx = mn_idx % N
    
    # Offsets for K dimension
    offs_k = tl.arange(0, BLOCK_SIZE)
    
    # Load in_4 for this (m, n) position across all K
    in_4_ptrs = in_4_ptr + (m_idx * stride_in4_m + n_idx * stride_in4_n + offs_k * stride_in4_k)
    mask_k = offs_k < K
    in_4 = tl.load(in_4_ptrs, mask=mask_k, other=0.0)
    
    # Load in_3 for this m position
    in_3_ptrs = in_3_ptr + (m_idx * stride_in3_m)
    in_3 = tl.load(in_3_ptrs, mask=mask_k, other=1.0)
    
    # Division
    tmp_3 = in_4 / in_3
    
    # Cast to float32
    tmp_4 = tmp_3.to(tl.float32)
    
    # Load in_0 for this (m, n) position
    in_0_ptrs = in_0_ptr + (m_idx * stride_in0_m + n_idx * stride_in0_n)
    in_0_val = tl.load(in_0_ptrs, mask=mask_k, other=0.0)
    
    # Unsqueeze broadcasts in_0 across K
    # Multiply with broadcasting
    tmp_6 = tmp_4 * in_0_val
    
    # Cast to float32 (tmp_7)
    tmp_7 = tmp_6.to(tl.float32)
    
    # Store tmp_7 result
    out_tmp7_ptrs = out_tmp7_ptr + (m_idx * stride_out_m + n_idx * stride_out_n + offs_k * stride_out_k)
    tl.store(out_tmp7_ptrs, tmp_7, mask=mask_k)
    
    # Compute LayerNorm along K dimension
    # First, compute mean
    sum_val = tl.sum(tmp_7, axis=0)
    mean = sum_val / K
    
    # Compute variance
    diff = tmp_7 - mean
    sq_diff = diff * diff
    var = tl.sum(sq_diff, axis=0) / K
    std = tl.sqrt(var + EPS)
    
    # Normalize
    normalized = diff / std
    
    # Load weight and bias
    weight = tl.load(in_2_ptr + offs_k * stride_in2_k, mask=mask_k, other=1.0)
    bias = tl.load(in_1_ptr + offs_k * stride_in1_k, mask=mask_k, other=0.0)
    
    # Apply weight and bias
    tmp_8 = normalized * weight + bias
    
    # Store tmp_8 result
    out_tmp8_ptrs = out_tmp8_ptr + (m_idx * stride_out_m + n_idx * stride_out_n + offs_k * stride_out_k)
    tl.store(out_tmp8_ptrs, tmp_8, mask=mask_k)


def triton_fused_all(in_0, in_1, in_2, in_3, in_4):
    """Complete wrapper: fused division, cast, multiply + layer_norm."""
    M = in_4.shape[0]
    N = in_4.shape[1]
    K = in_4.shape[2]
    
    total_positions = M * N
    
    # Output tensors
    out_tmp7 = torch.empty_like(in_4)
    out_tmp8 = torch.empty_like(in_4)
    
    # Define grid
    grid = (total_positions,)
    
    # Launch kernel
    fused_div_cast_mul_layernorm_kernel[grid](
        in_4, in_3, in_0, in_2, in_1, out_tmp7, out_tmp8,
        M, N, K,
        in_4.stride(0), in_4.stride(1), in_4.stride(2),
        in_3.stride(0),
        in_0.stride(0), in_0.stride(1),
        in_2.stride(0),
        in_1.stride(0),
        out_tmp7.stride(0), out_tmp7.stride(1), out_tmp7.stride(2),
        1e-05,
    )
    
    return out_tmp7, out_tmp8


# Pattern matching function - match the full computation including layer_norm
def pattern(in_0, in_1, in_2, in_3, in_4):
    """Match the full pattern including layer_norm."""
    # Division
    tmp_3 = in_4 / in_3
    # First cast
    tmp_4 = tmp_3.to(torch.float32)
    # Unsqueeze
    tmp_5 = in_0.unsqueeze(-1)
    # Multiply
    tmp_6 = tmp_4 * tmp_5
    # Second cast
    tmp_7 = tmp_6.to(torch.float32)
    # Layer norm
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (320,), in_2, in_1, 1e-05)
    # Return both outputs
    return tmp_7, tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract arguments needed for replacement."""
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    """Return the replacement function."""
    return triton_fused_all