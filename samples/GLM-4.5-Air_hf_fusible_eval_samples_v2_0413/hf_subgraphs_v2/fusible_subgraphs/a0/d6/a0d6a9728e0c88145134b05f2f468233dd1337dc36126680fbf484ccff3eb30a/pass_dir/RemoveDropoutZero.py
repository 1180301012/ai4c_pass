import torch
import triton
import triton.language as tl

def pattern(bmm_output):
    tmp_1 = torch.nn.functional.softmax(bmm_output, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    return tmp_1, tmp_2

def replacement_args(bmm_output):
    return (bmm_output,)

@triton.jit
def identity_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def identity_operator(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    identity_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    def drop_out_elimination(bmm_output):
        tmp_1 = torch.nn.functional.softmax(bmm_output, dim=-1)
        # Dropout with p=0.0 is identity operation
        return tmp_1, identity_operator(tmp_1)
    
    return drop_out_elimination