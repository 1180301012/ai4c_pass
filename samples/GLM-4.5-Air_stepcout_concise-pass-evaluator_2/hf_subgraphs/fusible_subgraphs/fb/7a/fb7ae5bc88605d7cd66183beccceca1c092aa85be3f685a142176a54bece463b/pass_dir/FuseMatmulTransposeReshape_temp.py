import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def matmul_transpose_reshape_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N_total, K,
    num_heads: tl.constexpr,
    N: tl.constexpr,
    stride_a0, stride_a1, stride_a2, stride_a3,
    stride_b0, stride_b1, stride_b2, stride_b3,
    stride_o0, stride_o1, stride_o2,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused matmul + transpose + reshape kernel.
    
    Input shapes:
        a: [batch, num_heads, M, K] = [1, 16, 257, 257]
        b: [batch, num_heads, K, N] = [1, 16, 257, 80]
    Output:
        [batch, M, num_heads * N] = [1, 257, 1280]
    """
    batch = tl.program_id(0)
    row = tl.program_id(1)
    
    # Initialize accumulator for each output column
    # We'll compute result[row, col] = sum over heads,k of a[head, row, k] * b[head, k, col-head*N]
    # where col = head * N + n_idx
    
    for col in range(N_total):
        head = col // N
        n_idx = col % N
        
        accumulator = 0.0
        
        # Iterate over K in blocks
        for k in range(0, K, BLOCK_SIZE):
            k_offsets = k + tl.arange(0, BLOCK_SIZE)
            k_mask = k_offsets < K
            
            # a: [batch, head, row, k] 
            a_offsets = (batch * stride_a0 + head * stride_a1 + 
                        row * stride_a2 + k_offsets * stride_a3)
            a_vals = tl.load(a_ptr + a_offsets, mask=k_mask, other=0.0)
            
            # b: [batch, head, k, n_idx]
            b_offsets = (batch * stride_b0 + head * stride_b1 + 
                        k_offsets * stride_b2 + n_idx * stride_b3)
            b_vals = tl.load(b_ptr + b_offsets, mask=k_mask, other=0.0)
            
            accumulator += tl.sum(a_vals * b_vals)
        
        # Store result
        out_offset = batch * stride_o0 + row * stride_o1 + col * stride_o2
        tl.store(out_ptr + out_offset, accumulator)


@torch.fx.wrap
def triton_matmul_transpose_reshape(a, b):
    """Fused matmul + transpose + reshape.
    
    Input:
        a: [1, num_heads, M, K] = [1, 16, 257, 257]
        b: [1, num_heads, K, N] = [1, 16, 257, 80]
    
    Output:
        [1, M, num_heads * N] = [1, 257, 1280]
    """
    batch, num_heads, M, K = a.shape
    N = b.shape[3]
    N_total = num_heads * N
    
    # Allocate output
    output = torch.empty((batch, M, N_total), dtype=a.dtype, device=a.device)
    
    # Grid: batch * M elements
    grid = (batch, M)
    
    matmul_transpose_reshape_kernel[grid](
        a, b, output,
        M, N_total, K,
        num_heads, N,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
    )
    
    return output


def pattern(in_0, in_1):
    """Match the pattern: matmul + transpose + contiguous + reshape + contiguous
    
    The original pattern:
        tmp_4 = tmp_3 @ in_1  # [1, 16, 257, 80]
        tmp_5 = tmp_4.transpose(1, 2)  # [1, 257, 16, 80]
        tmp_6 = tmp_5.contiguous()
        tmp_7 = tmp_6.reshape(1, 257, -1)  # [1, 257, 1280]
        tmp_8 = tmp_7.contiguous()
        return tmp_8
    """
    # First do the computation that we want to optimize
    tmp_4 = torch.matmul(in_0, in_1)  # This needs to be the softmax output
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.reshape(1, 257, -1)
    tmp_8 = tmp_7.contiguous()
    return tmp_8


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_matmul_transpose_reshape