import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Pattern for SmolLM3-3B: linear followed by element-wise multiply
    in_0: weight [out_features, in_features]
    in_1: input to linear [batch, seq, in_features]
    in_2: multiplier [batch, seq, out_features]
    """
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = in_2 * tmp_1
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_mul_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr, OUT_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for A (input)
    stride_am, stride_ak,
    # Strides for B (weight.T)
    stride_bk, stride_bn,
    # Strides for C (multiplier)
    stride_cm, stride_cn,
    # Strides for output
    stride_outm, stride_outn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused matmul + element-wise multiply kernel"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to first block
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop over K
    for k in range(0, K, BLOCK_K):
        # Masks
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Output mask
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load C for element-wise multiply
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=out_mask, other=0.0)
    
    # Fused multiply - convert accumulator to output dtype
    out = acc.to(c.dtype) * c
    
    # Store result
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, out, mask=out_mask)


@torch.fx.wrap
def fused_linear_mul(weight, input_tensor, multiplier):
    """
    Fused linear + element-wise multiply
    weight: [out_features, in_features]
    input_tensor: [batch, seq, in_features]
    multiplier: [batch, seq, out_features]
    """
    orig_shape = input_tensor.shape
    batch_seq = orig_shape[0] * orig_shape[1] if len(orig_shape) == 3 else orig_shape[0]
    in_features = orig_shape[-1]
    out_features = weight.shape[0]
    
    # Reshape input to 2D: [batch*seq, in_features]
    A = input_tensor.view(-1, in_features)
    # Weight transposed: [in_features, out_features]
    B = weight.t().contiguous()
    # Reshape multiplier to 2D: [batch*seq, out_features]
    C = multiplier.view(-1, out_features)
    
    M, K = A.shape
    K2, N = B.shape
    
    # Output tensor
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel
    fused_linear_mul_kernel[grid](
        A, B, C, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        out.stride(0), out.stride(1),
    )
    
    # Reshape output back to original batch shape
    out_shape = list(orig_shape[:-1]) + [out_features]
    return (out.view(*out_shape),)


def replacement_func():
    return fused_linear_mul