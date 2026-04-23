import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: linear followed by layer_norm with normalized_shape=(256,)
def pattern(in_8, in_7, in_6, in_2, in_3):
    linear = torch.nn.functional.linear(in_8, in_7, in_6)
    tmp_9 = torch.nn.functional.layer_norm(linear, (256,), in_3, in_2, 1e-05)
    return tmp_9

# Argument extraction function
# Returns all input tensors required for the fused operation
def replacement_args(in_8, in_7, in_6, in_2, in_3):
    return (in_8, in_7, in_6, in_2, in_3)


# Triton kernel for fused linear + layer_norm
@triton.jit
def fused_linear_layernorm_kernel(
    in_8_ptr, in_7_ptr, in_6_ptr, in_2_ptr, in_3_ptr,
    output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    
    start_m = tl.program_id(0) * BLOCK_M
    num_rows = tl.minimum(BLOCK_M, M - start_m)

    # Initialize accumulators for sum and sum of squares
    sum_val = tl.zeros((num_rows,), tl.float32)
    sum_sq = tl.zeros((num_rows,), tl.float32)

    # Intermediate storage for linear output (per row)
    linear_vals = tl.zeros((num_rows, BLOCK_N), tl.float32)

    # Process linear operation with matmul + bias
    for k in range(0, K, BLOCK_K):
        for n in range(0, N, BLOCK_N):
            # Load input blocks
            in_8_block = tl.load(
                in_8_ptr + start_m * K + k,
                shape=(num_rows, BLOCK_K),
                mask=(tl.arange(0, num_rows)[:, None] < M - start_m) & 
                      (tl.arange(0, BLOCK_K)[None, :] < K - k)
            )
            in_7_block = tl.load(
                in_7_ptr + k * N + n,
                shape=(BLOCK_K, BLOCK_N),
                mask=(tl.arange(0, BLOCK_K)[:, None] < K - k) & 
                      (tl.arange(0, BLOCK_N)[None, :] < N - n)
            )

            # Perform matmul and accumulate
            acc = tl.zeros((num_rows, BLOCK_N), tl.float32)
            for i in range(BLOCK_K):
                for j in range(BLOCK_N):
                    acc += in_8_block[:, i] * in_7_block[i, j]
            linear_vals += acc

            # Add bias for current n block
            bias_block = tl.load(
                in_6_ptr + n,
                shape=(BLOCK_N,),
                mask=(tl.arange(0, BLOCK_N) < N - n)
            )
            for i in range(num_rows):
                linear_vals[i, :tl.minimum(BLOCK_N, N - n)] += bias_block

            # Accumulate sum and sum of squares for layer_norm
            for i in range(num_rows):
                for j in range(tl.minimum(BLOCK_N, N - n)):
                    val = linear_vals[i, j]
                    sum_val[i] += val
                    sum_sq[i] += val * val

    # Compute mean and variance
    eps = 1e-05
    mean = sum_val / N
    var = (sum_sq - sum_val * sum_val / N) / N

    # Compute normalized output
    for i in range(num_rows):
        for j in range(N):
            val = linear_vals[i, j] - mean[i]
            den = tl.sqrt(var[i] + eps)
            normalized = val / den
            normalized = normalized * tl.load(in_3_ptr + j) + tl.load(in_2_ptr + j)
            tl.store(
                output_ptr + (start_m + i) * N + j,
                normalized
            )


# Kernel wrapper with proper tensor allocation
@torch.fx.wrap
def fused_linear_layernorm(in_8, in_7, in_6, in_2, in_3):
    M, _, K = in_8.shape
    N = in_7.shape[1]
    output = torch.empty((M, N), dtype=in_8.dtype, device=in_8.device)

    # Determine block sizes for optimal performance
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Compute grid configuration
    grid = (tl.cdiv(M, BLOCK_M),)

    # Launch kernel
    fused_linear_layernorm_kernel[grid](
        in_8,
        in_7,
        in_6,
        in_2,
        in_3,
        output,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return output

# Replacement function
# Returns the fused kernel function

def replacement_func():
    return fused_linear_layernorm