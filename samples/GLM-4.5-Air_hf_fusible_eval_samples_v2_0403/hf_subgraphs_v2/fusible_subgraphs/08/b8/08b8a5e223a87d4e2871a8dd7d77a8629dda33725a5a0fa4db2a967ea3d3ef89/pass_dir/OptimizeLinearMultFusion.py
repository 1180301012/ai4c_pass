import torch
import triton
import triton.language as tl

# Pattern for linear followed by multiplication (dependent operations)
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = in_1 * linear
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized fused kernel for linear followed by multiplication
@triton.jit
def optimized_fused_kernel(
    weight_ptr, in1_ptr, in2_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_FINAL: tl.constexpr,
):
    # Matrix dimensions
    stride_weight_col = N
    stride_weight_row = K
    stride_in1_col = K
    stride_in1_row = M
    stride_in2_flat = M * N
    stride_out_flat = M * N
    
    # Block offsets for matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_flat = tl.program_id(2)  # For final multiplication
    
    # Matrix multiplication phase
    m_block_start = pid_m * BLOCK_SIZE_M
    n_block_start = pid_n * BLOCK_SIZE_N
    
    offsets_m = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    # Accumulate matrix multiplication result in float32 for precision
    linear_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offsets_k < K
        
        # Load weights: [N, K] (transposed input)
        weight = tl.load(weight_ptr + offsets_n[:, None] * stride_weight_col + offsets_k[None, :] * stride_weight_row,
                         mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Load input: [M, K]
        input_vals = tl.load(in1_ptr + offsets_m[:, None] * stride_in1_col + offsets_k[None, :] * stride_in1_row,
                            mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Compute matrix multiplication: weight^T @ input
        linear_acc += tl.dot(weight.T, input_vals)
    
    # Convert to appropriate precision
    linear_result = linear_acc.to(tl.float16 if weight_ptr.dtype.element_ty == tl.float16 else tl.bfloat16)
    
    # Final multiplication phase
    flat_start = pid_flat * BLOCK_SIZE_FINAL
    offsets_flat = flat_start + tl.arange(0, BLOCK_SIZE_FINAL)
    mask_flat = offsets_flat < (M * N)
    
    # Flatten the linear result
    linear_flat = linear_result.reshape(-1)
    
    # Load operands for final multiplication
    linear_vals = tl.load(linear_flat + offsets_flat, mask=mask_flat, other=0.0)
    in2_vals = tl.load(in2_ptr + offsets_flat, mask=mask_flat, other=0.0)
    
    # Perform element-wise multiplication
    out_vals = linear_vals * in2_vals
    
    # Store final result
    tl.store(out_ptr + offsets_flat, out_vals, mask=mask_flat)

@torch.fx.wrap
def optimized_fused_linear_mult(in_0, in_1, in_2):
    # Get dimensions: Linear is [K, N] @ [M, K]^T -> [M, N]
    weight_shape = in_0.shape  # [K, N]
    in2_shape = in_2.shape      # [M, K]
    M, K, N = in2_shape[0], in2_shape[2], weight_shape[1]
    
    # Create output tensor
    out_shape = in_2.shape if in_2.dim() > 1 else (M, N)
    out = torch.empty(out_shape, device=in_1.device, dtype=in_1.dtype)
    
    total_elements = M * N
    
    if total_elements > 0 and K > 0:
        # Matrix multiplication configuration
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 16
        
        grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        # Final multiplication blocks
        BLOCK_SIZE_FINAL = 1024
        grid_final = (total_elements + BLOCK_SIZE_FINAL - 1) // BLOCK_SIZE_FINAL
        grid = (grid_m, grid_n, grid_final)
        
        optimized_fused_kernel[grid](
            in_0, in_2, in_1, out,
            M, N, K,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            BLOCK_SIZE_FINAL
        )
    
    return (out,)

def replacement_func():
    return optimized_fused_linear_mult