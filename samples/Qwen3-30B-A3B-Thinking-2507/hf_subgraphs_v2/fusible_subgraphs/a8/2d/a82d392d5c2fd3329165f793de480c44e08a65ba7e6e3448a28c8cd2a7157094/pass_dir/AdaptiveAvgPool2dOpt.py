import torch
import triton
import triton.language as tl


def pattern(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, (32, 24))

def replacement_args(x):
    return (x, 32, 24)


@triton.jit
def avg_pool_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    k = tl.program_id(0)
    n = k // C
    c = k % C
    x_base = n * C * H_in * W_in + c * H_in * W_in
    y_base = n * C * H_out * W_out + c * H_out * W_out
    i = tl.thread_id(0)
    j = tl.thread_id(1)
    if i >= H_out or j >= W_out:
        return
    input_i = i * 2
    input_j = j * 2
    x00 = tl.load(x_ptr + x_base + input_i * W_in + input_j)
    x01 = tl.load(x_ptr + x_base + input_i * W_in + input_j + 1)
    x10 = tl.load(x_ptr + x_base + (input_i + 1) * W_in + input_j)
    x11 = tl.load(x_ptr + x_base + (input_i + 1) * W_in + input_j + 1)
    avg = (x00 + x01 + x10 + x11) * 0.25
    tl.store(y_ptr + y_base + i * W_out + j, avg)


@torch.fx.wrap
def avg_pool2d_wrapper(x, H_out, W_out):
    N, C, H_in, W_in = x.shape
    H_out = 32
    W_out = 24
    y = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    grid = (N * C, 1)
    avg_pool_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        N=N,
        C=C,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        BLOCK_H=H_out,
        BLOCK_W=W_out
    )
    return y

def replacement_func():
    return avg_pool2d_wrapper