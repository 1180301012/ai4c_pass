import torch
import triton
import triton.language as tl

# Pattern matching function - matches just the norm operation
def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Triton kernel for L2 norm - outputs [M, 1] tensor
@triton.jit
def norm_kernel(
    in_ptr,
    out_ptr,
    M,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    # pid = row index
    pid = tl.program_id(0)
    
    # Compute sum of squares for this row
    sum_sq = 0.0
    for k in range(K):
        offset = pid * K + k
        x = tl.load(in_ptr + offset).to(tl.float32)
        sum_sq += x * x
    
    # Compute L2 norm
    norm = tl.sqrt(sum_sq + 1e-12)
    
    # Store result with shape [M, 1] - store at out_ptr + pid * 1
    tl.store(out_ptr + pid * 1, norm)


def norm_func(in_1):
    M, K = in_1.shape
    
    # Create output tensor [M, 1] using allowed API
    out = torch.empty((M, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    norm_kernel[(M,)](
        in_ptr=in_1,
        out_ptr=out,
        M=M,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return norm_func