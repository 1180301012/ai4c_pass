import torch
import triton
import triton.language as tl

def pattern(x):
    gelu_out = torch.nn.functional.gelu(x)
    mean_out = gelu_out.mean((2, 3), keepdim=True)
    return gelu_out, mean_out

def replacement_args(x):
    return (x,)

@triton.jit
def sum_reduction_kernel(in_ptr, out_ptr, B, C, H, W, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    b_idx = block_id // C
    c_idx = block_id % C
    in_offset = b_idx * C * H * W + c_idx * H * W
    out_offset = b_idx * C + c_idx

    shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    num_elements = H * W
    elements_per_thread = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    total = 0.0
    for i in range(elements_per_thread):
        pos = tl.thread_id(0) * elements_per_thread + i
        if pos < num_elements:
            total += tl.load(in_ptr + in_offset + pos)
    shared[tl.thread_id(0)] = total
    tl.sync()

    if tl.thread_id(0) == 0:
        sum_val = 0.0
        for i in range(BLOCK_SIZE):
            sum_val += shared[i]
        tl.store(out_ptr + out_offset, sum_val)

@torch.fx.wrap
def custom_mean(x):
    B, C, H, W = x.shape
    sum_tensor = torch.empty(B, C, 1, 1, dtype=x.dtype, device=x.device)
    grid = (B * C,)
    sum_reduction_kernel[grid](
        x,
        sum_tensor,
        B, C, H, W,
        BLOCK_SIZE=1024,
    )
    mean_tensor = sum_tensor / (H * W)
    return mean_tensor

@torch.fx.wrap
def replacement_wrapper(x):
    gelu_out = torch.nn.functional.gelu(x)
    mean_out = custom_mean(gelu_out)
    return gelu_out, mean_out

def replacement_func():
    return replacement_wrapper