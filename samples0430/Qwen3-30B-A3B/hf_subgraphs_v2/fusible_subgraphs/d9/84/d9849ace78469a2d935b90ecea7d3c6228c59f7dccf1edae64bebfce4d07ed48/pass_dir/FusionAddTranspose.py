import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    in_2 = in_1 + in_0
    tmp_2 = in_2.transpose(1, 2)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def add_transpose_kernel(
    in0_ptr: tl.pointer_type(tl.float16),
    in1_ptr: tl.pointer_type(tl.float16),
    out_ptr: tl.pointer_type(tl.float16),
    N2: tl.int32,
    N3: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N2 * N3
    j = offsets // N3
    i = offsets % N3
    in1_idx = i * N2 + j
    in0_idx = i
    out_idx = j * N3 + i
    x = tl.load(in1_ptr + in1_idx, mask=mask)
    y = tl.load(in0_ptr + in0_idx, mask=mask)
    out = x + y
    tl.store(out_ptr + out_idx, out, mask=mask)

@torch.fx.wrap
def fused_add_transpose(in_0, in_1):
    N2 = in_1.shape[2]  # 19
    N3 = in_1.shape[1]  # 128
    total_size = N2 * N3
    out = torch.empty([1, N2, N3], dtype=in_1.dtype, device=in_1.device)
    BLOCK_SIZE = 128
    num_blocks = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_transpose_kernel[(num_blocks,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        N2=N2,
        N3=N3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_add_transpose