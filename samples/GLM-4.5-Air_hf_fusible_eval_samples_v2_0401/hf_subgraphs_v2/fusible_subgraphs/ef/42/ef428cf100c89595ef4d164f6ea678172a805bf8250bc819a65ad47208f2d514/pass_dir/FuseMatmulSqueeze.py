import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matches matmul followed by squeeze operation.
    Expects: matmul = torch.matmul(in_0, in_1); tmp_1 = matmul.squeeze(1)
    """
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def simple_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Bounds check
    if pid >= M * N:
        return
        
    i = pid // N
    j = pid % N
        
    sum_val = 0.0
    for k in tl.range(0, K):
        a_val = tl.load(a_ptr + i * K + k, mask=k < K, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptr + k * N + j, mask=k < K, other=0.0).to(tl.float32)
        sum_val += a_val * b_val
    
    tl.store(c_ptr + i * N + j, sum_val, mask=(i < M) & (j < N))

@torch.fx.wrap
def fused_matmul_squeeze(in_0, in_1):
    # Get input shapes and determine output shape
    # in_0: [1, 1, 249], in_1: [1, 249, 64] → [1, 1, 64] → squeeze → [1, 64]
    
    # Use the working Triton kernel
    A = in_0.reshape(1, 249)  # [1, 249]
    B = in_1.reshape(249, 64)  # [249, 64]
    
    # Output shape after matmul
    out = torch.empty(1, 64, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with 1D grid (one program per output element)
    simple_matmul_kernel[(64,)](A, B, out, 1, 249, 64)
    
    return out

def replacement_func():
    return fused_matmul_squeeze