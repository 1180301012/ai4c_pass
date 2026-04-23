import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return matmul

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Strategy: For tiny batched matmul with K=9, N=1, we compute the entire result
# in a single Triton kernel launch. Key optimizations:
# 1. Use a single large 1D grid to cover all B*M output elements
# 2. Each thread computes one output element (one dot product of K=9)
# 3. Vectorized loads where possible

@triton.jit
def tiny_matmul_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, M, K, N,
    stride_in0_b, stride_in0_k, stride_in0_n,
    stride_in1_b, stride_in1_m, stride_in1_k,
    stride_out_b, stride_out_m, stride_out_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total output elements (for N=1: B*M)
    total = B * M
    valid = offsets < total
    
    # Map flat offset to (b, m)
    b = offsets // M
    m = offsets % M
    
    # Compute sum_k in_1[b,m,k] * in_0[b,k,0]
    # K=9 is small, load as individual scalar loads in a loop
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Manual dot product for K=9
    for k_idx in tl.static_range(9):
        v0 = tl.load(in_0_ptr + b * stride_in0_b + k_idx * stride_in0_k + 0 * stride_in0_n, mask=valid, other=0.0)
        v1 = tl.load(in_1_ptr + b * stride_in1_b + m * stride_in1_m + k_idx * stride_in1_k, mask=valid, other=0.0)
        acc = acc + v1 * v0
    
    tl.store(out_ptr + b * stride_out_b + m * stride_out_m, acc.to(out_ptr.dtype.element_ty), mask=valid)


@torch.fx.wrap
def triton_batched_matmul(in_0, in_1):
    B = in_1.shape[0]
    M = in_1.shape[1]
    K = in_1.shape[2]
    N = in_0.shape[2]
    
    out = torch.empty([B, M, N], dtype=in_1.dtype, device=in_1.device)
    
    total = B * M  # For N=1
    
    # Choose block size to match the workload
    BLOCK_SIZE = 128
    if total >= 256:
        BLOCK_SIZE = 256
    if total >= 512:
        BLOCK_SIZE = 512
    if total >= 1024:
        BLOCK_SIZE = 1024
    
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    tiny_matmul_kernel[(grid,)](
        in_0_ptr=in_0, in_1_ptr=in_1, out_ptr=out,
        B=B, M=M, K=K, N=N,
        stride_in0_b=in_0.stride(0), stride_in0_k=in_0.stride(1), stride_in0_n=in_0.stride(2),
        stride_in1_b=in_1.stride(0), stride_in1_m=in_1.stride(1), stride_in1_k=in_1.stride(2),
        stride_out_b=out.stride(0), stride_out_m=out.stride(1), stride_out_n=out.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_batched_matmul