import torch
import triton
import triton.language as tl

def pattern(hidden_states, weight_matrix, bias):
    # Linear transformation: must match exactly what's in model.py
    linear = torch.nn.functional.linear(hidden_states, weight_matrix, bias)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    return linear, tmp_5, tmp_6

def replacement_args(hidden_states, weight_matrix, bias):
    return (hidden_states, weight_matrix, bias)

@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    m = pid // grid_n
    n = pid % grid_n

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Pointer arithmetic for matrix multiplication
    x_ptr += m * BLOCK_SIZE_M * K
    y_ptr += n * BLOCK_SIZE_N
    out_ptr = out_ptr + m * BLOCK_SIZE_M * N + n * BLOCK_SIZE_N
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, K)
        
        # Load x and y tiles
        x = tl.load(x_ptr + (k//BLOCK_SIZE_K)*BLOCK_SIZE_M*BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_M)[:, None]*K + tl.arange(k, k_end)[None, :], mask=tl.arange(k, k_end)[None, :] < K, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + (k//BLOCK_SIZE_K)*BLOCK_SIZE_N*K + tl.arange(k, k_end)[:, None]*K + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=tl.arange(k, k_end)[:, None] < K, other=0.0).to(tl.float32)
        
        # Matrix multiplication
        acc += tl.sum(x[:, :, None] * y[None, :, :], axis=1)
    
    # Load bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(n * BLOCK_SIZE_N, (n + 1) * BLOCK_SIZE_N), mask=tl.arange(BLOCK_SIZE_N) < ((n + 1) * BLOCK_SIZE_N - n * BLOCK_SIZE_N), other=0.0).to(tl.float32)
        acc += bias
    
    # Store result
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :] * M + tl.arange(BLOCK_SIZE_M)[:, None], acc, mask=tl.arange(BLOCK_SIZE_M)[:, None] < BLOCK_SIZE_M)

@torch.fx.wrap
def optimized_linear_view_transpose(hidden_states, weight_matrix, bias):
    # Use Triton kernel for matrix multiplication instead of torch.nn.functional.linear
    M, N, K = hidden_states.size(0), hidden_states.size(2), weight_matrix.size(0)
    output = torch.empty((M, hidden_states.size(1), N), dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Launch Triton kernel
    grid_size = (M * N + 63) // 64
    matmul_kernel[grid_size](
        x_ptr=hidden_states,
        y_ptr=weight_matrix,
        bias_ptr=bias,
        out_ptr=output,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32
    )
    
    # View and transpose operations
    viewed_output = output.view(1, 1, -1, 64)
    transposed_output = viewed_output.transpose(1, 2)
    
    return output, viewed_output, transposed_output

def replacement_func():
    return optimized_linear_view_transpose