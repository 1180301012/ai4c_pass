import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Just match matmul + scalar multiplication first
    tmp_0 = torch.matmul(in_2, in_1)
    tmp_1 = tmp_0 * in_0
    return (tmp_1,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_kernel(
    a_ptr,      # in_2: [2, 512] 
    b_ptr,      # in_1: [512, 1]
    scale_val,  # in_0: scalar
    out_ptr,    # tmp_1: [2, 1]
    M: tl.constexpr,  # 2
    N: tl.constexpr,  # 512  
    K: tl.constexpr,  # 1
):
    # Since this is a small fixed-size computation, we handle all rows in one program
    row_idx = tl.program_id(0)
    
    # Initialize accumulator for this row
    acc = 0.0
    
    # Loop over columns of A (rows of B) to compute dot product
    for k in range(N):
        # Load element from A: [row_idx, k]
        a_offset = row_idx * N + k
        a = tl.load(a_ptr + a_offset, mask=a_offset < (M * N), other=0.0)
        
        # Load element from B: [k, 0] (since K=1)
        b_offset = k * K + 0
        b = tl.load(b_ptr + b_offset, mask=b_offset < (N * K), other=0.0)
        
        # Accumulate product
        acc += a * b
    
    # Scale by the scalar value
    result = acc * scale_val
    
    # Store result for this row
    out_offset = row_idx * 1 + 0
    tl.store(out_ptr + out_offset, result, mask=(row_idx) < M)

@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    M, N = in_2.shape
    K = in_1.shape[1]
    
    # Create output tensor
    tmp_1 = torch.empty((M, K), device=in_2.device, dtype=in_2.dtype)  # [2, 1]
    
    # Launch kernel - one program per row
    grid = (M,)
    
    fused_kernel[grid](
        in_2,
        in_1, 
        in_0.item(),  # Extract scalar value
        tmp_1,
        M, N, K
    )
    
    return tmp_1

def replacement_func():
    return fused_matmul_scale