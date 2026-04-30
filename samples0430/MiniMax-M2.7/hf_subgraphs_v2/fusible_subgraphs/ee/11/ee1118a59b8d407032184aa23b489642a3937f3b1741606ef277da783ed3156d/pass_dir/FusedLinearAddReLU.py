import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: Linear + Add + ReLU
    - torch.nn.functional.linear(in_3, in_1, in_0) computes: in_3 @ in_1^T + in_0
    - in_2 + linear is element-wise addition
    - tmp_3.relu_() is ReLU activation
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused kernel.
    - in_0: bias tensor [128]
    - in_1: weight tensor [128, 128]  
    - in_2: add tensor [1000, 128]
    - in_3: input tensor [1000, 128]
    """
    return (in_0, in_1, in_2, in_3)


# Optimized fused kernel using 2D blocking with shared memory
@triton.jit
def fused_linear_add_relu_kernel(
    # Pointers
    in_0_ptr,  # bias [N]
    in_1_ptr,  # weight [N, K]
    in_2_ptr,  # add tensor [M, N]
    in_3_ptr,  # input tensor [M, K]
    out_ptr,   # output [M, N]
    # Shapes
    M, N, K,
    # Strides
    stride_in_1_k, stride_in_1_n,
    stride_in_2_n, stride_in_2_m,
    stride_in_3_k, stride_in_3_m,
    stride_out_n, stride_out_m,
    # Block sizes - must be constexpr for tl.arange
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: out = relu(in_2 + in_3 @ in_1 + in_0)
    Uses vectorized 2D blocking optimized for [1000, 128] x [128, 128] matmul.
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load bias for this N block
    bias = tl.load(in_0_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        # Update K offsets and mask for this block
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        
        # Load a BLOCK_K-sized chunk of in_3: [BLOCK_M, BLOCK_K]
        x_ptrs = in_3_ptr + offs_m[:, None] * stride_in_3_m + offs_k[None, :] * stride_in_3_k
        x_mask = mask_m[:, None] & mask_k[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load a BLOCK_K-sized chunk of in_1: [BLOCK_N, BLOCK_K]
        w_ptrs = in_1_ptr + offs_n[:, None] * stride_in_1_n + offs_k[None, :] * stride_in_1_k
        w_mask = mask_n[:, None] & mask_k[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Accumulate: x @ w.T -> [BLOCK_M, BLOCK_N]
        accumulator += tl.dot(x, tl.trans(w))
    
    # Add bias
    result = accumulator + bias[None, :]
    
    # Load in_2 for addition
    in_2_data = tl.load(
        in_2_ptr + offs_m[:, None] * stride_in_2_m + offs_n[None, :] * stride_in_2_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    )
    
    # Add in_2 and apply ReLU
    result = tl.where(result + in_2_data > 0, result + in_2_data, 0.0)
    
    # Store output
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
        result,
        mask=mask_m[:, None] & mask_n[None, :]
    )


@torch.fx.wrap
def fused_linear_add_relu(in_0, in_1, in_2, in_3):
    """
    Wrapper function for the fused Linear + Add + ReLU kernel.
    """
    M, K = in_3.shape
    N = in_0.shape[0]
    
    # Allocate output tensor
    out = torch.empty((M, N), dtype=in_3.dtype, device=in_3.device)
    
    # Fixed block sizes - optimized for this problem
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Calculate 2D grid
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    # Get strides
    stride_in_1_k = in_1.stride(1) if len(in_1.stride()) > 1 else 1
    stride_in_1_n = in_1.stride(0)
    stride_in_2_n = in_2.stride(1) if len(in_2.stride()) > 1 else 1
    stride_in_2_m = in_2.stride(0)
    stride_in_3_k = in_3.stride(1) if len(in_3.stride()) > 1 else 1
    stride_in_3_m = in_3.stride(0)
    stride_out_n = out.stride(1) if len(out.stride()) > 1 else 1
    stride_out_m = out.stride(0)
    
    # Launch kernel with constexpr block sizes
    fused_linear_add_relu_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        M=M, N=N, K=K,
        stride_in_1_k=stride_in_1_k,
        stride_in_1_n=stride_in_1_n,
        stride_in_2_n=stride_in_2_n,
        stride_in_2_m=stride_in_2_m,
        stride_in_3_k=stride_in_3_k,
        stride_in_3_m=stride_in_3_m,
        stride_out_n=stride_out_n,
        stride_out_m=stride_out_m,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out


def replacement_func():
    return fused_linear_add_relu