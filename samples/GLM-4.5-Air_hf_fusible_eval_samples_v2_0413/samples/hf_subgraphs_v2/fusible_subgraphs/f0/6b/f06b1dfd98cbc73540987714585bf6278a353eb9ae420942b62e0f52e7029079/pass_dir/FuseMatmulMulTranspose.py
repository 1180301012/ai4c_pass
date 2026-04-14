import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # The computation to match - exactly as in model.py
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.t()
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_matmul_mul_transpose_kernel(
    out_1_ptr,  # output for tmp_1 [2, 1]
    out_2_ptr,  # output for tmp_2 [1, 2] 
    in_0_val,   # scalar value for multiplication
    in_1_ptr,   # input in_1 [1024, 1]
    in_2_ptr,   # input in_2 [2, 1024]
    M: tl.constexpr,  # 2 (rows of in_2)
    K: tl.constexpr,  # 1024 (inner dimension)
    N: tl.constexpr,  # 1 (cols of in_1)
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute matrix C = A @ B where A = in_2 [M, K], B = in_1 [K, N], C = [M, N]
    
    # Each program handles one row of the output matrix
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    if row >= M or col >= N:
        return
    
    # Compute dot product for position [row, col] using original data type
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute bounds for this block
        block_k = min(BLOCK_SIZE_K, K - k)
        if k + block_k > K:
            block_k = K - k
        
        # Load block from A [row, k:k+block_k]
        a_ptr = in_2_ptr + row * K + k
        mask_a = tl.arange(0, block_k) < block_k
        a_block = tl.load(a_ptr + tl.arange(0, block_k), mask=mask_a, other=0.0)
        
        # Load block from B [k:k+block_k, col]
        b_ptr = in_1_ptr + k * N + col
        mask_b = tl.arange(0, block_k) < block_k
        b_block = tl.load(b_ptr + tl.arange(0, block_k), mask=mask_b, other=0.0)
        
        # Accumulate dot product
        for i in range(block_k):
            sum_val += a_block[i] * b_block[i]
    
    # Apply scalar multiplication and cast back to original data type
    result_val = sum_val * in_0_val
    out_1_ptr_addr = out_1_ptr + row * N + col
    tl.store(out_1_ptr_addr, result_val.to(tl.float32), boundary_check=True)
    
    # Store transposed result for tmp_2
    out_2_ptr_addr = out_2_ptr + col * M + row
    tl.store(out_2_ptr_addr, result_val.to(tl.float32), boundary_check=True)

@torch.fx.wrap
def fused_matmul_mul_transpose(in_0, in_1, in_2):
    M, K = in_2.shape
    K_in1, N = in_1.shape
    
    # Create output tensors
    out_1 = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)
    out_2 = torch.empty((N, M), dtype=in_2.dtype, device=in_2.device)
    
    # Get scalar value - handle both scalar tensor and actual scalar
    if in_0.dim() == 0:
        in_0_val = in_0.item()
    else:
        # This should be a scalar tensor with shape []
        in_0_val = in_0.flatten()[0].item()
    
    # Block size for K dimension - use smaller blocks for better performance on small matrices
    BLOCK_SIZE_K = 64
    
    # Calculate grid size - only need M x N programs since we're processing each output element
    grid = (M, N, 1)
    
    # Launch kernel
    fused_matmul_mul_transpose_kernel[grid](
        out_1_ptr=out_1,
        out_2_ptr=out_2,
        in_0_val=float(in_0_val),  # Ensure it's a Python float
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        M=M,
        K=K,
        N=N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out_1, out_2

def replacement_func():
    return fused_matmul_mul_transpose