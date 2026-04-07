import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - matches exactly one device transfer AND matmul
def pattern(in_0, in_1, in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    tmp_3 = in_1.to(device(type='cuda'))
    tmp_4 = in_0.to(device(type='cuda'))
    return (tmp_4, tmp_3, matmul)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for matrix multiplication - optimized for small matrices
@triton.jit
def matmul_kernel_small(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    K,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    output_dtype: tl.constexpr,
):
    # Program ID 
    pid = tl.program_id(0)
    
    # Number of programs for dimension M
    num_pid_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_M * num_pid_N
    
    # Get row/col of program
    group_id = pid // num_pid_in_group
    first_pid_M = group_id * num_pid_M
    
    group_size_M = min(num_pid_M, 2048)  # Limit group size
    row_group = (pid % num_pid_in_group) // group_size_M
    pid_in_group = pid % num_pid_in_group
    
    col_group = (pid_in_group % group_size_M)
    M_block = row_group
    N_block = col_group
    
    row_offset = M_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offset = N_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Compute row and column offsets
    row_offsets = row_offset[:, None]
    col_offsets = col_offset[None, :]
    
    x_ptrs = x_ptr + row_offsets * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    y_ptrs = y_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * N + col_offsets
    
    # Use float32 for accumulation as it has better precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(row_offsets[:, None] < M) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0.0)
        y = tl.load(y_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & (col_offsets[None, :] < N), other=0.0)
        
        accumulator += tl.dot(x, y)
        x_ptrs += BLOCK_SIZE_K
        y_ptrs += BLOCK_SIZE_K * N

    # Convert to appropriate output dtype based on the input dtype
    if output_dtype == torch.float16:
        accumulator = accumulator.to(tl.float16)
    elif output_dtype == torch.bfloat16:
        accumulator = accumulator.to(tl.bfloat16)
    
    out_offsets = row_offsets[:, None] * N + col_offsets
    out_ptrs = out_ptr + out_offsets
    tl.store(out_ptrs, accumulator, mask=(row_offsets[:, None] < M) & (col_offsets[None, :] < N))

# Combined function that batches device transfers and optimized matmul
@torch.fx.wrap
def batched_device_transfer_matmul(in_0, in_1, in_2, in_3):
    # Batch the device transfers for better performance
    in_0_cuda = in_0.to(device(type='cuda'))
    in_1_cuda = in_1.to(device(type='cuda'))
    
    # Get tensor properties for optimized matmul
    x_shape = in_2.shape
    y_shape = in_3.shape
    
    M, K = x_shape
    _, N = y_shape
    
    # Create output tensor
    output_dtype = in_2.dtype
    out = torch.empty((M, N), dtype=output_dtype, device='cuda')
    
    # Choose optimal block size based on tensor sizes
    if M <= 64 and N <= 64:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    elif M <= 128 and N <= 128:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 32, 32
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
    
    # Calculate grid size
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # Launch the optimized matmul kernel with dtype
    matmul_kernel_small[grid](
        in_2,
        in_3,
        out,
        M,
        K,
        N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        output_dtype
    )
    
    return in_0_cuda, in_1_cuda, out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return batched_device_transfer_matmul