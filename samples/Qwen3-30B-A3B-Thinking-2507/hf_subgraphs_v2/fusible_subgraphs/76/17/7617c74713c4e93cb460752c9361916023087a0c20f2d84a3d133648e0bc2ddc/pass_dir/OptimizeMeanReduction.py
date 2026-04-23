import torch
import triton
import triton.language as tl

def pattern(x):
    return x.mean((2, 3), keepdim=False)

def replacement_args(x):
    return (x, )

@triton.jit
def mean_reduction_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    b_id = block_idx // C
    c_id = block_idx % C
    input_offset = b_id * C * H * W + c_id * H * W
    out_offset = b_id * C + c_id
    sum = 0.0
    for i in range(H):
        for j in range(W):
            idx = input_offset + i * W + j
            sum += tl.load(x_ptr + idx)
    mean = sum / (H * W)
    tl.store(out_ptr + out_offset, mean)

@torch.fx.wrap
def mean_reduction_wrapper(x):
    B, C, H, W = x.shape
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    num_blocks = B * C
    BLOCK_SIZE = 32
    mean_reduction_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return mean_reduction_wrapper