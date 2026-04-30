import torch
import triton
import triton.language as tl

"""
Pass to fuse dropout (p=0.0, training=False) + type_cast + linear into single kernel.
For RECT_L models where dropout is a no-op.
"""

def pattern(in_0, in_1, in_2):
    """
    Match pattern: dropout(in_2, p=0.0, training=False) -> to(dtype) -> linear(...)
    """
    # Dropout with p=0.0 is essentially a no-op when training=False
    # Match the exact pattern from RECT_L model
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fusion_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused dropout(no-op) + type_cast + linear kernel.
    Since dropout is a no-op, we just do the linear operation.
    """
    # Block indices
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    group_size_n = num_pid_n
    
    # Within-group position
    pid_m = (pid % num_pid_in_group) // num_pid_n
    pid_n = pid % num_pid_n
    
    # Bounds
    m_start = first_pid_m + pid_m
    n_start = pid_n
    m_step = 1
    n_step = 1
    
    # Iterate over output tiles
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load weight tile
        w_offsets = (
            (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_wk) +
            (tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_wn)
        )
        w_mask = (
            (tl.arange(0, BLOCK_SIZE_K)[:, None] < K) &
            (tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
        )
        w = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0)
        
        # Load input tile
        i_offsets = (
            (tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_im) +
            ((k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :]) * stride_ik
        )
        i_mask = (
            (tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
            ((k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] < K)
        )
        inp = tl.load(input_ptr + i_offsets, mask=i_mask, other=0.0)
        
        # Matmul
        acc += tl.dot(inp, w)
        
        # Advance weight pointer (keep weight stationary)
    
    # Add bias if present
    if bias_ptr != 0:
        bias_offsets = tl.arange(0, BLOCK_SIZE_N)
        bias_mask = bias_offsets < N
        bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc = acc + bias[None, :]
    
    # Output tile bounds
    m = m_start * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = n_start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = m < M
    n_mask = n < N
    
    # Convert to output dtype and store
    out = acc.to(tl.float16)
    out_ptrs = (
        (m[:, None] * N) + n[None, :]
    )
    tl.store(output_ptr + out_ptrs, out, mask=(m_mask[:, None] & n_mask[None, :]))


@torch.fx.wrap
def fusion_wrapper(in_0, in_1, in_2):
    """
    Wrapper for the fused dropout + linear kernel.
    Dropout with p=0.0 is a no-op, so we just perform the linear operation.
    """
    # Input shape: [128, 128], Weight shape: [128, 128], Bias shape: [128]
    M = in_2.shape[0]  # 128
    K = in_2.shape[1]  # 128
    N = in_1.shape[0]  # 128
    
    # Ensure inputs are contiguous
    in_2_contig = in_2.contiguous()
    in_1_contig = in_1.contiguous()
    in_0_contig = in_0.contiguous()
    
    # Output
    output = torch.empty((M, N), dtype=torch.float16, device=in_2.device)
    
    # Block sizes - tune for 128x128 matmul
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Grid size
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_programs = grid_m * grid_n
    
    fusion_kernel[(num_programs,)](
        input_ptr=in_2_contig,
        weight_ptr=in_1_contig,
        bias_ptr=in_0_contig,
        output_ptr=output,
        M=M, K=K, N=N,
        stride_im=in_2.stride(0), stride_ik=in_2.stride(1),
        stride_wn=in_1.stride(0), stride_wk=in_1.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output


def replacement_func():
    return fusion_wrapper