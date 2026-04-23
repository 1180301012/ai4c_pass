import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    return matmul

def replacement_args(in_2, in_3):
    return (in_2, in_3)

@triton.jit
def matmul_vec_kernel(
    A_ptr, B_ptr, C_ptr,
    M,
    stride_am, stride_an,
    stride_bn,
    stride_cm, stride_ck,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    a_row_ptr = A_ptr + row_idx * stride_am
    acc = 0.0
    
    for block_start in range(0, M, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        a_vals = tl.load(a_row_ptr + offsets * stride_an, mask=mask, other=0.0).to(tl.float32)
        b_vals = tl.load(B_ptr + offsets * stride_bn, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a_vals * b_vals)
    
    c_ptr = C_ptr + row_idx * stride_cm
    tl.store(c_ptr, acc)


@torch.fx.wrap
def matmul_vec(A, B):
    rows = A.shape[0]
    M = A.shape[1]
    cols = B.shape[1]
    
    # Allocate output directly with known shape
    C = torch.empty(rows, cols, dtype=A.dtype, device=A.device)
    
    stride_am, stride_an = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_ck = C.stride()
    
    grid = (rows,)
    
    matmul_vec_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=C,
        M=M,
        stride_am=stride_am, stride_an=stride_an,
        stride_bn=stride_bn,
        stride_cm=stride_cm, stride_ck=stride_ck,
        BLOCK_SIZE=256,
        num_warps=1,
        num_stages=1,
    )
    
    return C

def replacement_func():
    return matmul_vec