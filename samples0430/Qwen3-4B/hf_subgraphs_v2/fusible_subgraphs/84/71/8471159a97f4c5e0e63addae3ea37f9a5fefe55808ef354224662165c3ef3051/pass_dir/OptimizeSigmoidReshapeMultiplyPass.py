import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    sig = torch.sigmoid(linear)
    sig_reshaped = sig.view(sig.shape[0], 64, 1, 1)
    return in_3 * sig_reshaped

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_kernel(
in_0_ptr,
in_1_ptr,
in_2_ptr,
in_3_ptr,
out_ptr,
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    for i in range(block_start, block_start + BLOCK_SIZE):
        if i >= B:
            break
        in_2_i = tl.load(in_2_ptr + i * N)
        in_1_i = tl.load(in_1_ptr + i * M * N)
        linear_out = tl.dot(in_2_i, in_1_i) + tl.load(in_0_ptr + i * M)
        sig = tl.sigmoid(linear_out)
        tl.store(out_ptr + i * (M * C * H), tl.load(in_3_ptr + i * (M * C * H)) * sig)

@torch.fx.wrap
def optimized_wrapper(in_0, in_1, in_2, in_3):
    B = in_2.shape[0]
    M = 64
    N = 8
    C = in_3.shape[2]
    H = in_3.shape[3]
    out = torch.empty_like(in_3)
    optimized_kernel[tl.cdiv(B, 128)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        B=B,
        M=M,
        N=N,
        C=C,
        H=H,
        BLOCK_SIZE=128
    )
    return out

def replacement_func():
    return optimized_wrapper