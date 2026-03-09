import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = tmp_0 * in_1
    tmp_3 = torch.as_tensor(tmp_1, device=torch.device('cuda'))
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def scalar_mul_cuda_kernel(result_ptr, scalar_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    if BLOCK_SIZE == 1:
        # For scalar case, just store the result at offset 0
        mask = offset == 0
        tl.store(result_ptr + offset, scalar_val, mask=mask)
    else:
        # For vectorized case, broadcast scalar to all elements
        mask = offset < 1
        tl.store(result_ptr + offset, scalar_val, mask=mask)

@torch.fx.wrap
def fused_scalar_mul_cuda(in_0, in_1):
    # Perform scalar multiplication and directly create CUDA tensor
    result = torch.as_tensor([in_0 * in_1], dtype=torch.int64, device='cuda')
    return result

def replacement_func():
    return fused_scalar_mul_cuda