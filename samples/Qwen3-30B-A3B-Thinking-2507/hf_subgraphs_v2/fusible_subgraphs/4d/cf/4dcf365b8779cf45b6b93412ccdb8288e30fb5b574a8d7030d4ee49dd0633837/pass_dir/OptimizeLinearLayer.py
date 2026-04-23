import torch
import triton
import triton.language as tl

def pattern(x, w, b):
    # Match the linear layer operation
    return torch.nn.functional.linear(x, w, b)

def replacement_args(x, w, b):
    # Return the necessary arguments for the kernel
    return (x, w, b)

@triton.jit
def linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, K, M, BLOCK_SIZE_M: tl.constexpr):
    # Each program handles a block of M elements
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_SIZE_M
    m_offsets = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M

    # Load bias for the current block
    bias = tl.load(b_ptr + m_offsets, mask=m_mask)

    # Load input into shared memory (N x K)
    # Assuming N is small enough for shared memory (N <= 32)
    input_buf = tl.zeros((N, K), dtype=tl.float32)
    for n in range(N):
        # Load input row n from global memory
        input_row = tl.load(x_ptr + n * K + tl.arange(0, K), mask=(tl.arange(0, K) < K))
        input_buf[n, :] = input_row

    # Compute output for each element in the current M block
    for n in range(N):
        acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for k in range(K):
            # Load weight element (M x K)
            w_val = tl.load(w_ptr + m_offsets[:, None] * K + k, mask=m_mask[:, None], other=0.0)
            # Load input element
            x_val = input_buf[n, k]
            # Multiply and accumulate
            acc += w_val * x_val
        # Add bias and store result
        out = acc + bias
        tl.store(out_ptr + n * M + m_offsets, out, mask=m_mask)

@torch.fx.wrap
def optimized_linear(x, w, b):
    # Calculate dimensions
    N = x.numel() // x.shape[-1]  # Batch * sequence dimension
    K = x.shape[-1]
    M = w.shape[0]  # Output features

    # Configure kernel
    BLOCK_SIZE_M = 128
    num_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # Allocate output
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)

    # Launch kernel
    linear_kernel[(num_blocks,)](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        N=N,
        K=K,
        M=M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    # Reshape output to match input dimensions
    return out.view(x.shape[:-1] + (M,))

def replacement_func():
    return optimized_linear