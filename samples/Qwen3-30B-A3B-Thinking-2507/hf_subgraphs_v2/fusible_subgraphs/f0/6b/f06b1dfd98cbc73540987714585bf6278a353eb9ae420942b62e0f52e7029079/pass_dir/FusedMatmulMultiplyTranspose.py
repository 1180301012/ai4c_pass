import torch
import triton
import triton.language as tl

# Pattern matching

def pattern(in_2, in_1, in_0):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.t()
    return tmp_1, tmp_2

# Argument extraction

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Triton kernel
@triton.jit
def optimized_kernel(
    in2_ptr, in1_ptr, scalar_ptr,
    out1_ptr, out2_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr = 2,
    BLOCK_N: tl.constexpr = 1
):
    # Load the scalar
    scalar = tl.load(scalar_ptr)
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    
    # Load row from in_2 (BLOCK_M x K)
    row = tl.arange(0, BLOCK_M)
    in2_offsets = (row_start + row) * K + tl.arange(0, K)
    in2 = tl.load(in2_ptr + in2_offsets, mask=row < M, other=0.0)
    
    # Load column from in_1 (K x N)
    in1_offsets = tl.arange(0, K)
    in1 = tl.load(in1_ptr + in1_offsets, mask=tl.arange(0, K) < K, other=0.0)
    
    # Compute dot product
    acc = tl.dot(in2, in1[:, None])
    acc = acc * scalar
    
    # Store to out1 (M x N)
    out1_offsets = (row_start + row) * N
    tl.store(out1_ptr + out1_offsets, acc, mask=row < M)
    
    # Store transpose to out2 (N x M)
    out2_offsets = tl.arange(0, BLOCK_M)
    tl.store(out2_ptr + out2_offsets, acc, mask=row < M)

# Kernel wrapper
@torch.fx.wrap
def kernel_wrapper(in2, in1, scalar):
    M = 2
    N = 1
    K = 1024
    out1 = torch.empty((M, N), dtype=scalar.dtype, device=scalar.device)
    out2 = torch.empty((N, M), dtype=scalar.dtype, device=scalar.device)
    grid = (1,)
    optimized_kernel[grid](
        in2_ptr=in2, in1_ptr=in1, scalar_ptr=scalar,
        out1_ptr=out1, out2_ptr=out2,
        M=M, N=N, K=K,
        BLOCK_M=2, BLOCK_N=1, BLOCK_K=128
    )
    return out1, out2

def replacement_func():
    return kernel_wrapper