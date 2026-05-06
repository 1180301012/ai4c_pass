import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    t1 = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    t2 = t1 + in_3
    t3 = t2 * in_0
    t4 = t3 + in_2
    return t4.contiguous()

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def optimize_einsum_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    B,
    C,
    H,
    J,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    b_id = tl.program_id(0)
    c_id = tl.program_id(1)
    h_id = tl.program_id(2)
    w_id = tl.program_id(3)
    
    total = tl.zeros(1, tl.float32)
    for j in range(J):
        
        a_val = tl.load(a_ptr + (b_id * B + c_id * C + h_id * H + j) * J)
        b_val = tl.load(b_ptr + (b_id * B + h_id * W + w_id * J + j))
        total += a_val * b_val
    
    tl.store(out_ptr + (b_id * B + c_id * C + h_id * H + w_id), total)

@torch.fx.wrap
def optimize_einsum(a, b, c, d, e):
    B = a.shape[0]
    C = a.shape[1]
    H = a.shape[2]
    J = a.shape[3]
    W = b.shape[3]
    out = torch.empty(B, C, H, W, dtype=a.dtype, device=a.device)
    
    optimize_einsum_kernel[(B, 1, 1, 1)](a, b, out, B, C, H, J, W, BLOCK_SIZE=128)
    
    return out

def replacement_func():
    return optimize_einsum