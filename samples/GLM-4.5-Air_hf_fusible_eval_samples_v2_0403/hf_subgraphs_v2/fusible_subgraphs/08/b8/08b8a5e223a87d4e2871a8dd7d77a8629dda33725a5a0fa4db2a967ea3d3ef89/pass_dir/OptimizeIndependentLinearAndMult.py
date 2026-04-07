import torch
import triton
import triton.language as tl

# Pattern for independent linear and multiplication operations
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return (tmp_3, linear)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernels for linear and multiplication
@triton.jit
def optimized_linear_kernel(
    weight_ptr, in_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix dimensions
    stride_am = M
    stride_ak = K
    stride_bn = N
    stride_bk = K
    stride_cm = M
    stride_cn = N
    
    # Block offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_block_start = pid_m * BLOCK_SIZE_M
    n_block_start = pid_n * BLOCK_SIZE_N
    
    # Create offsets within block
    offsets_m = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < K
        
        # Load weights
        weight = tl.load(weight_ptr + offsets_n[:, None] * stride_bn + offsets_k[None, :] * stride_bk,
                         mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Load input
        input_vals = tl.load(in_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
                            mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Compute and accumulate
        acc += tl.dot(weight.T, input_vals)
    
    # Store result
    out = acc.to(tl.float16 if weight_ptr.dtype.element_ty == tl.float16 else tl.bfloat16)
    tl.store(out_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn,
             out, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def optimized_mult_kernel(
    in1_ptr, in2_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (M * N)
    
    # Load inputs
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    in2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    
    # Compute element-wise multiplication
    out = in1 * in2
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_linear_and_mult(in_0, in_1, in_2, in_3):
    # Linear operation dimensions: [K, N] @ [M, K]^T -> [M, N]
    weight_shape = in_0.shape  # [K, N]
    in_shape = in_3.shape      # [M, K]
    M, K, N = in_shape[0], in_shape[2], weight_shape[1]
    
    # Create output tensors
    linear_out = torch.empty((M, N), device=in_3.device, dtype=in_3.dtype)
    mult_out = torch.empty_like(in_2)
    
    # Launch linear kernel
    if N > 0 and K > 0 and M > 0:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        grid = (grid_m, grid_n)
        
        optimized_linear_kernel[grid](
            in_0, in_3, linear_out,
            M, N, K,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
    
    # Launch multiplication kernel
    mult_elements = in_2.numel()
    if mult_elements > 0:
        BLOCK_SIZE = 1024
        grid = (mult_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        optimized_mult_kernel[grid](
            in_2, in_1, mult_out,
            mult_elements, 1,
            BLOCK_SIZE
        )
    
    return (mult_out, linear_out)

def replacement_func():
    return optimized_linear_and_mult