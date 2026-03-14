import torch
import triton
import triton.language as tl

def pattern(in_5):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    return tmp_6

def replacement_args(in_5):
    return (in_5,)

@triton.jit
def adaptive_avg_pool2d_kernel(
    in_ptr, out_ptr,
    N, C, H, W, total_elements
):
    pid = tl.program_id(0)
    
    if pid >= total_elements:
        return
    
    idx_N = pid // C
    idx_C = pid % C
    
    sum_val = 0.0
    count = 0
    
    for h in range(H):
        for w in range(W):
            in_idx = idx_N * C * H * W + idx_C * H * W + h * W + w
            val = tl.load(in_ptr + in_idx)
            sum_val += val
            count += 1
    
    avg_val = sum_val / count if count > 0 else 0.0
    tl.store(out_ptr + pid, avg_val)

@torch.fx.wrap
def optimized_adaptive_avg_pool2d(in_5):
    N, C, H, W = in_5.shape
    
    out = torch.empty((N, C, 1, 1), device=in_5.device, dtype=in_5.dtype)
    
    total_elements = N * C
    
    adaptive_avg_pool2d_kernel[(total_elements,)](
        in_ptr=in_5,
        out_ptr=out,
        N=N, C=C, H=H, W=W, total_elements=total_elements
    )
    
    return out

def replacement_func():
    return optimized_adaptive_avg_pool2d