import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min = 1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Autotune different configurations
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def optimized_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    stride_0, stride_1, k_dim,
    n_elements, n_dim,
):
    pid = tl.program_id(0)
    
    # Compute position from pid
    row_idx = pid // k_dim
    col_idx = pid % k_dim
    
    # Initialize accumulators
    weighted_sum = 0.0
    sum_factor = 0.0
    
    # Reduction loop
    for i in range(n_dim):
        idx = row_idx * stride_0 + i * stride_1 + col_idx
        in_0_val = tl.load(in_0_ptr + idx).to(tl.float32)
        in_1_val = tl.load(in_1_ptr + idx).to(tl.float32)
        weighted_sum = weighted_sum + in_1_val * in_0_val
        sum_factor = sum_factor + in_0_val

    # Clamp and divide
    result = weighted_sum / tl.maximum(sum_factor, 1e-09)
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    M, N, K = in_0.shape  # [1, 10, 1024]
    
    # Make tensors contiguous and get actual strides
    in_0_c = in_0.contiguous()
    in_1_c = in_1.contiguous()
    
    stride_0 = in_0_c.stride(0)  # N * K = 10240
    stride_1 = in_0_c.stride(1)  # K = 1024
    
    num_programs = M * K  # 1024
    
    out = torch.empty((M, K), dtype=torch.float32, device=in_0.device)
    
    optimized_kernel[(num_programs,)](
        in_0_c, in_1_c, out,
        stride_0, stride_1, K,
        num_programs,
        N,
    )
    
    return out

def replacement_func():
    return kernel_wrapper