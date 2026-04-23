import torch
import triton
import triton.language as tl


@triton.jit
def fused_view_mul_kernel(
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    n_start = pid_n * BLOCK_N
    k_start = pid_k * BLOCK_K
    
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    
    n_mask = n_offsets < N
    k_mask = k_offsets < K
    
    # Load in_1 (1D tensor [N]) - broadcast across K dimension
    in_1_vals = tl.load(in_1_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Load in_2 (2D tensor [N, K])
    in_2_ptrs = in_2_ptr + n_offsets * K + k_offsets
    in_2_mask = n_mask & k_mask
    in_2_vals = tl.load(in_2_ptrs, mask=in_2_mask, other=0.0)
    
    # Compute broadcast multiply: in_1.view(-1, 1) * in_2
    out_vals = in_1_vals * in_2_vals
    
    # Store result
    out_ptrs = out_ptr + n_offsets * K + k_offsets
    tl.store(out_ptrs, out_vals, mask=in_2_mask)


@torch.fx.wrap
def dispatch_fused_view_mul(in_0, in_1, in_2, route):
    N = in_1.shape[0]
    K = in_2.shape[1]
    
    # Allocate output for the multiply result
    out = torch.empty((N, K), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_N = 32
    BLOCK_K = 32
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_k = (K + BLOCK_K - 1) // BLOCK_K
    
    fused_view_mul_kernel[(grid_n, grid_k)](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        N=N,
        K=K,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    # Compute tmp_3: in_0.view(-1,1).expand_as(out)
    tmp_3 = in_0.view((-1, 1)).expand_as(out)
    
    # Compute tmp_4: zeros tensor based on route
    if route == "zeros_1000_16":
        tmp_4 = out.new_zeros((1000, 16))
    elif route == "zeros_128_128":
        tmp_4 = out.new_zeros((128, 128))
    else:
        tmp_4 = out.new_zeros((1000, 16))
    
    return (tmp_3, tmp_4, out)


# Pattern for (1000, 16) variants (bfloat16 and float32)
def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    tmp_4 = tmp_1.new_zeros((1000, 16))
    return (tmp_3, tmp_4, tmp_1)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "zeros_1000_16")

def replacement_func():
    return dispatch_fused_view_mul