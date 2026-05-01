import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    matmul = torch.matmul(in_1, in_0)
    return torch.reshape(matmul, [-1, 16])

def replacement_args(in_1, in_0):
    return (in_1, in_0, 16)

@triton.jit
def matmul_reshape_kernel(
    a_ptr, b_ptr, out_ptr,
    B, H_out, K, D, N,
    stride_a_b, stride_a_h, stride_a_k,
    stride_b_b, stride_b_k, stride_b_d,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_end = min(k_start + BLOCK_K, K)
        
        # Load A block (shape [BLOCK_M, BLOCK_K])
        a = tl.load(
            a_ptr + block_m * stride_a_b + k_start * stride_a_k + 
            tl.arange(0, BLOCK_M)[:, None] * stride_a_b + 
            tl.arange(0, BLOCK_K)[None, :] * stride_a_k,
            mask=(block_m + tl.arange(0, BLOCK_M)[:, None] < B) & 
                  (k_start + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0
        )

        # Load B block (shape [BLOCK_M, BLOCK_K])
        b = tl.load(
            b_ptr + block_m * stride_b_b + k_start * stride_b_k + 
            tl.arange(0, BLOCK_M)[:, None] * stride_b_b + 
            tl.arange(0, BLOCK_K)[None, :] * stride_b_k,
            mask=(block_m + tl.arange(0, BLOCK_M)[:, None] < B) & 
                  (k_start + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0
        )

        # Perform matmul for this block
        acc += tl.dot(a, b)

    # Calculate output indices (m, n) -> (i, j, d)
    m_idx = block_m + tl.arange(0, BLOCK_M)[:, None]
    n_idx = block_n + tl.arange(0, BLOCK_N)[None, :]
    
    # Total elements = B * H_out * D
    # M = (B * H_out * D) // N
    # Element index in flattened matmul = m_idx * N + n_idx
    # Convert to (i, j, d):
    #   i = (m_idx * N + n_idx) // (H_out * D)
    #   rest = (m_idx * N + n_idx) % (H_out * D)
    #   j = rest // D
    #   d = rest % D
    
    # Instead, we write to the flattened output directly
    out = out_ptr + block_m * stride_out_m + block_n * stride_out_n
    tl.store(out, acc, mask=(block_m + tl.arange(0, BLOCK_M)[:, None] < B * H_out * D // N) & 
                                      (block_n + tl.arange(0, BLOCK_N)[None, :] < N))

@torch.fx.wrap
def kernel_wrapper(in_1, in_0, N):
    B, H_out, K = in_1.shape
    B, K, D = in_0.shape
    M = (B * H_out * D) // N  # Must be integer division
    
    out = torch.empty((M, N), dtype=in_1.dtype, device=in_1.device)
    
    stride_a_b, stride_a_h, stride_a_k = in_1.stride()
    stride_b_b, stride_b_k, stride_b_d = in_0.stride()
    stride_out_m, stride_out_n = out.stride()
    
    # Block sizes for Triton kernel
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    matmul_reshape_kernel[(grid_m, grid_n)](
        in_1, in_0, out,
        B, H_out, K, D, N,
        stride_a_b, stride_a_h, stride_a_k,
        stride_b_b, stride_b_k, stride_b_d,
        stride_out_m, stride_out_n,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return out

def replacement_func():
    return kernel_wrapper