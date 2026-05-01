import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp0 = torch.nn.functional.silu(in_1, inplace=True)
    result = tmp0 + in_0
    return result

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def silu_add_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Compute SiLU: x * sigmoid(x)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-in1))
    silu = in1 * sigmoid_val

    out = silu + in0
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def silu_add(in0, in1):
    N = in0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in0)
    silu_add_kernel[(num_programs,)](
        in0_ptr=in0,
        in1_ptr=in1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return silu_add