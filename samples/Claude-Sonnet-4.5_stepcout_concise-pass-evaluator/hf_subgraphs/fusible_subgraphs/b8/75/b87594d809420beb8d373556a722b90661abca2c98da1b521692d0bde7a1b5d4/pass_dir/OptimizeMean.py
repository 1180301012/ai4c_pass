import torch
import triton
import triton.language as tl


# Pattern matching function - matches mean operation
def pattern(in_0):
    # in_0 is the input tensor [N, S, K]
    tmp = in_0.mean(-2)
    return tmp


def replacement_args(in_0):
    return (in_0,)


# Triton kernel for mean reduction along dim -2 - optimized with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_S': 16}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_S': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_S': 49}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_S': 64}, num_stages=3, num_warps=4),
    ],
    key=['S'],
)
@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    N, S, K,
    stride_in_n, stride_in_s, stride_in_k,
    stride_out_n, stride_out_k,
    BLOCK_SIZE_S: tl.constexpr
):
    # Each program computes one (n, k) position in output
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Initialize sum
    sum_val = 0.0
    
    # Sum over sequence dimension
    for s in range(0, S, BLOCK_SIZE_S):
        s_offsets = s + tl.arange(0, BLOCK_SIZE_S)
        mask_s = s_offsets < S
        
        # Load input [BLOCK_SIZE_S]
        inp_ptrs = input_ptr + row_idx * stride_in_n + s_offsets * stride_in_s + col_idx * stride_in_k
        inp = tl.load(inp_ptrs, mask=mask_s, other=0.0)
        
        sum_val += tl.sum(inp)
    
    # Compute mean
    mean_val = sum_val / S
    
    # Store result
    out_ptr = output_ptr + row_idx * stride_out_n + col_idx * stride_out_k
    tl.store(out_ptr, mean_val)


@torch.fx.wrap
def triton_mean(input_tensor):
    """Optimized mean reduction using Triton"""
    # Get shapes
    N = input_tensor.shape[0]  # batch size
    S = input_tensor.shape[1]  # sequence length
    K = input_tensor.shape[2]  # features
    
    # Allocate output tensor
    mean_out = torch.empty((N, K), dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel
    mean_kernel[(N, K)](
        input_tensor, mean_out,
        N, S, K,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        mean_out.stride(0), mean_out.stride(1),
    )
    
    return mean_out


def replacement_func():
    return triton_mean