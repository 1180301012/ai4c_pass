import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp = linear.permute(0, 3, 1, 2)
    return tmp

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_linear_permute_kernel(
    in3_ptr,
    in1_ptr,
    in0_ptr,
    out_ptr,
    N1, N2, N3, N4,
    C_out, C_in,
    BLOCK_SIZE: tl.constexpr
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    i_j = tl.thread_id(0)
    i_k = tl.thread_id(1)
    j = block_id_j * BLOCK_SIZE + i_j
    k = block_id_k * BLOCK_SIZE + i_k
    
    if j >= N2 or k >= N3:
        return
    
    for i in tl.arange(0, C_out):
        acc = tl.zeros((1,), dtype=tl.float32)
        for m in tl.arange(0, C_in):
            in3_val = tl.load(in3_ptr + (0 * N2 * N3 * N4 + j * N3 * N4 + k * N4 + m))
            in1_val = tl.load(in1_ptr + (i * C_in + m))
            acc = acc + in3_val * in1_val
        bias_val = tl.load(in0_ptr + i)
        out_val = acc + bias_val
        out_idx = i * (N2 * N3) + j * N3 + k
        tl.store(out_ptr + out_idx, out_val)

@torch.fx.wrap
def fused_linear_permute(in3, in1, in0):
    batch_size, n2, n3, n4 = in3.shape
    c_out, c_in = in1.shape
    out = torch.empty((batch_size, c_out, n2, n3), dtype=in3.dtype, device=in3.device)
    
    grid_x = (n2 + 64 - 1) // 64
    grid_y = (n3 + 64 - 1) // 64
    
    fused_linear_permute_kernel[(grid_x, grid_y), 64*64](
        in3,
        in1,
        in0,
        out,
        batch_size, n2, n3, n4,
        c_out, c_in,
        BLOCK_SIZE=64
    )
    
    return out

def replacement_func():
    return fused_linear_permute