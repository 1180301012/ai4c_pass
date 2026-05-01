import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    H,
    W,
    C,
    BLOCK_SIZE: tl.constexpr=64
):
    total_elements = C * H * W
    block_id = tl.program_id(0)
    start = block_id * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, total_elements)
    for i in range(start, end):
        c = i // (H * W)
        hw = i % (H * W)
        h = hw // W
        w = hw % W
        x = tl.load(in_2_ptr + c)
        inv = 1.0 / (1.0 + tl.exp(-x))
        sig_val = inv
        in_0_val = tl.load(in_0_ptr + i)
        in_1_val = tl.load(in_1_ptr + i)
        result = in_1_val * sig_val + in_0_val
        tl.store(out_ptr + i, result)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    C = in_1.shape[1]
    H = in_1.shape[2]
    W = in_1.shape[3]
    assert in_2.shape == (1, 1, C), f"Expected in_2 shape (1, 1, {C}), got {in_2.shape}"
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 64
    num_blocks = (C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    optimized_kernel[(num_blocks,)](
        in_0,
        in_1,
        in_2,
        out,
        H,
        W,
        C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return kernel_wrapper