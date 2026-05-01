import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    split = torch.functional.split(tmp_4, [38, 57, 57], dim=1)
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    return tmp_6, tmp_7, tmp_8

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def split_kernel(in_ptr, out0_ptr, out1_ptr, out2_ptr, H: tl.constexpr, W: tl.constexpr):
    block_id = tl.program_id(0)
    i = block_id // W
    j = block_id % W

    # Calculate base pointers for input and outputs
    in_base = in_ptr + i * 152 * W + j
    out0_base = out0_ptr + i * 38 * W + j
    out1_base = out1_ptr + i * 57 * W + j
    out2_base = out2_ptr + i * 57 * W + j

    # Write 38 channels to out0
    for c in range(38):
        tl.store(out0_base + c * W, tl.load(in_base + c * W))

    # Write 57 channels to out1 (offset 38)
    for c in range(57):
        tl.store(out1_base + c * W, tl.load(in_base + (38 + c) * W))

    # Write 57 channels to out2 (offset 95)
    for c in range(57):
        tl.store(out2_base + c * W, tl.load(in_base + (95 + c) * W))

@torch.fx.wrap
def split_kernel_wrapper(tmp_4):
    _, _, H, W = tmp_4.shape
    out0 = torch.empty((1, 38, H, W), dtype=tmp_4.dtype, device=tmp_4.device)
    out1 = torch.empty((1, 57, H, W), dtype=tmp_4.dtype, device=tmp_4.device)
    out2 = torch.empty((1, 57, H, W), dtype=tmp_4.dtype, device=tmp_4.device)

    grid = (H * W,)
    split_kernel[grid](
        tmp_4, out0, out1, out2, H, W
    )

    return out0, out1, out2

def replacement_func():
    return split_kernel_wrapper