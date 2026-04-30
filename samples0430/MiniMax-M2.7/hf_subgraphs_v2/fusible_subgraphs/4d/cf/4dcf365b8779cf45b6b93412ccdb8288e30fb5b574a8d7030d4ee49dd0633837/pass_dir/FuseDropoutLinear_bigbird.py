import torch
import triton
import triton.language as tl

"""
Pass to fuse dropout (p=0.1) + linear into single optimized kernel.
For BigBird RoBERTa models.
"""

def pattern(in_0, in_1, in_2):
    """
    Match pattern: dropout(in_2, 0.1, False, False) -> linear(...)
    """
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def dropout_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    p: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused dropout + linear kernel.
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
    
    # Create dropout mask
    off_m = m_start * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = n_start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    
    # Random mask for dropout (generate one value per row for efficiency)
    rng_offset = pid * 17  # Arbitrary offset for randomness
    rand_vals = tl.rand(
        (rng_offset & 0xFFFFFFFF).to(tl.uint32),
        ((rng_offset // 1024) & 0xFFFFFFFF).to(tl.uint32)
    )
    keep_mask = rand_vals > p
    
    # Iterate over K dimension
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
            (off_m[:, None] * stride_im) +
            ((k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :]) * stride_ik
        )
        i_mask = (
            (off_m[:, None] < M) &
            ((k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] < K)
        )
        inp = tl.load(input_ptr + i_offsets, mask=i_mask, other=0.0)
        
        # Apply dropout mask
        inp = inp * keep_mask.to(tl.float32)
        
        # Matmul
        acc += tl.dot(inp, w)
    
    # Scale by keep probability for training correctness
    scale = 1.0 / (1.0 - p)
    acc = acc * scale
    
    # Add bias if present
    if bias_ptr != 0:
        bias_offsets = tl.arange(0, BLOCK_SIZE_N)
        bias_mask = bias_offsets < N
        bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
        acc = acc + bias[None, :]
    
    # Convert to output dtype
    out = acc.to(tl.float32)
    
    # Store
    out_ptrs = (
        (off_m[:, None] * N) + off_n[None, :]
    )
    tl.store(output_ptr + out_ptrs, out, mask=mask)


@torch.fx.wrap
def fusion_wrapper(in_0, in_1, in_2):
    """
    Wrapper for the fused dropout + linear kernel.
    """
    # Input shape: [1, 17, 768], Weight shape: [3072, 768], Bias shape: [3072]
    # Linear: output[b, m, n] = sum_k(input[b, m, k] * weight[n, k]) + bias[n]
    M = in_2.shape[0] * in_2.shape[1]  # 1 * 17 = 17
    K = in_2.shape[2]  # 768
    N = in_1.shape[0]  # 3072
    
    # Reshape input for batched matmul: [1, 17, 768] -> [17, 768]
    in_2_reshaped = in_2.view(M, K)
    
    # Ensure inputs are contiguous
    in_2_contig = in_2_reshaped.contiguous()
    in_1_contig = in_1.contiguous()
    in_0_contig = in_0.contiguous()
    
    # Output: [17, 3072]
    output = torch.empty((M, N), dtype=torch.bfloat16, device=in_2.device)
    
    # Block sizes
    BLOCK_SIZE_M = 1  # Small M dimension
    BLOCK_SIZE_N = 64  # Larger N for parallelism
    BLOCK_SIZE_K = 32
    
    # Grid size
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_programs = grid_m * grid_n
    
    dropout_linear_kernel[(num_programs,)](
        input_ptr=in_2_contig,
        weight_ptr=in_1_contig,
        bias_ptr=in_0_contig,
        output_ptr=output,
        M=M, K=K, N=N,
        p=0.1,
        stride_im=in_2_contig.stride(0), stride_ik=in_2_contig.stride(1),
        stride_wn=in_1_contig.stride(0), stride_wk=in_1_contig.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output


def replacement_func():
    return fusion_wrapper