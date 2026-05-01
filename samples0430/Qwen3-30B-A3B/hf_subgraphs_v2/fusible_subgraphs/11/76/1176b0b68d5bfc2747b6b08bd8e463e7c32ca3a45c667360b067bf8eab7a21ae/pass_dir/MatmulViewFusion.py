import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    matmul = in_1 @ in_0
    tmp_1 = matmul.view(24, 128, 20, 20)  # Match the most common view shape pattern
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    # Extract the view shape (24, 128, 20, 20) from the pattern
    view_shape = (24, 128, 20, 20)
    return (in_0, in_1, view_shape)

# Triton kernel for batched matrix multiplication
@triton.jit
def batched_matmul_kernel(
    in1_ptr,
    in0_ptr,
    out_ptr,
    B: tl.int32,
    C: tl.int32,
    M: tl.int32,
    K: tl.int32,
    N: tl.int32,
    stride_in1_b: tl.int32,
    stride_in1_c: tl.int32,
    stride_in1_m: tl.int32,
    stride_in1_k: tl.int32,
    stride_in0_b: tl.int32,
    stride_in0_c: tl.int32,
    stride_in0_k: tl.int32,
    stride_in0_n: tl.int32,
    stride_out_b: tl.int32,
    stride_out_c: tl.int32,
    stride_out_m: tl.int32,
    stride_out_n: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    n_block_idx = tl.program_id(2)
    
    b = batch_idx // C
    c = batch_idx % C
    
    m_offset = m_block_idx * BLOCK_M
    n_offset = n_block_idx * BLOCK_N
    m_offsets = m_offset + tl.arange(0, BLOCK_M)
    n_offsets = n_offset + tl.arange(0, BLOCK_N)
    mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    
    for k in range(0, K, BLOCK_K):
        in1_offs = b * stride_in1_b + c * stride_in1_c + m_offset * stride_in1_m + k * stride_in1_k
        in1_block = tl.load(
            in1_ptr + in1_offs,
            mask=(m_offsets[:, None] < M) & (tl.arange(0, BLOCK_K) < BLOCK_K) & (k + tl.arange(0, BLOCK_K) < K),
            other=0.0
        )
        
        in0_offs = b * stride_in0_b + c * stride_in0_c + k * stride_in0_k + n_offset * stride_in0_n
        in0_block = tl.load(
            in0_ptr + in0_offs,
            mask=(tl.arange(0, BLOCK_K) < BLOCK_K) & (n_offsets[None, :] < N) & (k + tl.arange(0, BLOCK_K) < K),
            other=0.0
        )
        
        acc += tl.dot(in1_block, in0_block, allow_tf32=True)
    
    out_offs = b * stride_out_b + c * stride_out_c + m_offset * stride_out_m + n_offset * stride_out_n
    tl.store(out_ptr + out_offs, acc, mask=mask)

# Kernel wrapper
def matmul_view_fusion(in_0, in_1, view_shape):
    B = in_1.shape[0]
    C = in_1.shape[1]
    M = in_1.shape[2]
    K = in_1.shape[3]
    N = in_0.shape[3]
    
    out = torch.empty(view_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Get strides for the tensor
    stride_in1_b = in_1.stride(0)
    stride_in1_c = in_1.stride(1)
    stride_in1_m = in_1.stride(2)
    stride_in1_k = in_1.stride(3)
    stride_in0_b = in_0.stride(0)
    stride_in0_c = in_0.stride(1)
    stride_in0_k = in_0.stride(2)
    stride_in0_n = in_0.stride(3)
    stride_out_b = out.stride(0)
    stride_out_c = out.stride(1)
    stride_out_m = out.stride(2)
    stride_out_n = out.stride(3)
    
    # Set block sizes for Triton
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    num_batches = B * C
    
    batched_matmul_kernel[(num_batches, num_m_blocks, num_n_blocks)](
        in_1,
        in_0,
        out,
        B,
        C,
        M,
        K,
        N,
        stride_in1_b,
        stride_in1_c,
        stride_in1_m,
        stride_in1_k,
        stride_in0_b,
        stride_in0_c,
        stride_in0_k,
        stride_in0_n,
        stride_out_b,
        stride_out_c,
        stride_out_m,
        stride_out_n,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )
    
    return out

@torch.fx.wrap
def wrapper(in_0, in_1, view_shape):
    return matmul_view_fusion(in_0, in_1, view_shape)

# Replacement function
def replacement_func():
    return wrapper