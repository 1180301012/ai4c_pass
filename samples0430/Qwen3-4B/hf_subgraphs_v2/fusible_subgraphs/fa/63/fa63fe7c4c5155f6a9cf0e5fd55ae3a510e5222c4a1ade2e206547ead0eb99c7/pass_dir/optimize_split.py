import torch
import triton
import triton.language as tl

def pattern(in_1, in_0, in_2):
    matmul = in_1 @ in_0
    tmp_1 = in_1[..., 1:]
    tmp_2 = in_2[..., 1:]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 152, 7, 7)
    splits = torch.functional.split(tmp_4, [38, 57, 57], dim=1)
    split0 = splits[0]
    split1 = splits[1]
    split2 = splits[2]
    return (matmul, split0, split1, split2, tmp_1)

def replacement_args(in_1, in_0, in_2):
    return (in_1, in_0, in_2)

@triton.jit
def split_kernel(x_ptr, out0_ptr, out1_ptr, out2_ptr, size_x, split_sizes, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= size_x:
            break
        if idx < split_sizes[0]:
            tl.store(out0_ptr + idx, tl.load(x_ptr + idx))
        elif idx < split_sizes[0] + split_sizes[1]:
            tl.store(out1_ptr + idx, tl.load(x_ptr + idx))
        else:
            tl.store(out2_ptr + idx, tl.load(x_ptr + idx))

def kernel_wrapper(in_1, in_0, in_2):
    out0 = torch.empty_like(in_1)
    out1 = torch.empty_like(in_1)
    out2 = torch.empty_like(in_1)
    
    split_kernel[(1,)](
        x_ptr=in_1,
        out0_ptr=out0,
        out1_ptr=out1,
        out2_ptr=out2,
        size_x=in_1.numel(),
        split_sizes=(38, 57, 57),
        BLOCK_SIZE=256,
    )
    return (in_0, out0, out1, out2, in_1)

def replacement_func():
    return kernel_wrapper