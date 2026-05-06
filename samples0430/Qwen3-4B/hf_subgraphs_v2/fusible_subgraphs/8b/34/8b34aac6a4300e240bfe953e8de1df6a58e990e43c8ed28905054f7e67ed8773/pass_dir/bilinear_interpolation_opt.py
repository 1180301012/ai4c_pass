import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.interpolate(x, (64, 64), None, 'bilinear', False)

def replacement_args(x):
    return (x,)

@triton.jit
def bilinear_interp_kernel(
    input_ptr: tl.float32,
    output_ptr: tl.float32,
    N: tl.int32,
    C: tl.int32,
    H_in: tl.int32,
    W_in: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.store(output_ptr + pid, 0.0)

def bilinear_interp_wrapper(x):
    N = x.shape[0]
    C = x.shape[1]
    H_in = x.shape[2]
    W_in = x.shape[3]
    H_out = 64
    W_out = 64
    output = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    grid = (tl.cdiv(H_out, 16), tl.cdiv(W_out, 16))
    bilinear_interp_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        N=N,
        C=C,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        BLOCK_SIZE=16
    )
    return output

def replacement_func():
    return bilinear_interp_wrapper