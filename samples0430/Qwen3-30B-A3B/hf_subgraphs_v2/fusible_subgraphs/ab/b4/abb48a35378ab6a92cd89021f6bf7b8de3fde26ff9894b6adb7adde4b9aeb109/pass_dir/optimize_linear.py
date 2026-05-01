import torch
import triton
import triton.language as tl

# Constants for kernel
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32

@triton.jit
def linear_kernel(
    in_ptr, weight_ptr, out_ptr,
    in_row_stride, in_col_stride,
    weight_row_stride, weight_col_stride,
    out_row_stride, out_col_stride,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load input block
        in_block = tl.load(
            in_ptr + row_start * in_row_stride + k * in_col_stride,
            shape=(BLOCK_M, BLOCK_K),
            mask=(tl.arange(0, BLOCK_M)[:, None] < M - row_start) & (tl.arange(0, BLOCK_K)[None, :] < K - k),
            other=0.0
        )
        
        # Load weight block
        weight_block = tl.load(
            weight_ptr + k * weight_row_stride + col_start * weight_col_stride,
            shape=(BLOCK_K, BLOCK_N),
            mask=(tl.arange(0, BLOCK_K)[:, None] < K - k) & (tl.arange(0, BLOCK_N)[None, :] < N - col_start),
            other=0.0
        )
        
        # Perform the dot product
        acc += tl.dot(in_block, weight_block)
    
    # Store the result
    tl.store(
        out_ptr + row_start * out_row_stride + col_start * out_col_stride,
        acc,
        mask=(tl.arange(0, BLOCK_M)[:, None] < M - row_start) & (tl.arange(0, BLOCK_N)[None, :] < N - col_start)
    )

def optimized_linear(in_0, in_1):
    # in_1 shape: [batch, seq_len, in_features]
    batch, seq_len, in_features = in_1.shape
    out_features = in_0.shape[0]
    
    # The input for matrix multiplication will be [seq_len, in_features]
    # Weight will be [out_features, in_features]
    
    # Calculate grid dimensions
    grid_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_features + BLOCK_N - 1) // BLOCK_N
    
    # Create output tensor
    output = torch.empty(seq_len, out_features, dtype=in_1.dtype, device=in_1.device)
    
    # Set up kernel parameters
    linear_kernel[(grid_m, grid_n)](
        in_1.view(seq_len, in_features),  # [seq_len, in_features]
        in_0,  # [out_features, in_features]
        output,  # [seq_len, out_features]
        in_1.stride(1), in_1.stride(2),  # input strides
        in_0.stride(0), in_0.stride(1),  # weight strides
        output.stride(0), output.stride(1),  # output strides
        seq_len, out_features, in_features,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    # Reshape back to [batch, seq_len, out_features]
    return output.view(batch, seq_len, out_features)

def pattern(in_0, in_1):
    result = torch.nn.functional.linear(in_1, in_0, None)
    return result

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return optimized_linear