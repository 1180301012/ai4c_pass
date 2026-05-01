import torch
import triton
import triton.language as tl

def pattern(in0, in1, in2):
    tmp0 = in0 / 8.0
    tmp1 = tmp0 + in2
    tmp2 = tmp1 + in1
    return (tmp2,)

def replacement_args(in0, in1, in2):
    return (in0, in1, in2)

@triton.jit
def fused_kernel(x_ptr, in2_ptr, in1_ptr, out_ptr, n_elements, BLOCK_SIZE):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute 4D coordinates for batch (2), head (12), i (7), j (7)
    batch = offsets // (12 * 7 * 7)  # 588 elements per batch
    rest = offsets % (12 * 7 * 7)
    head = rest // (7 * 7)  # 49 elements per head
    i = (rest % (7 * 7)) // 7
    j = rest % 7

    # Compute in1 index: [batch, 0, 0, j] → flat index = batch * 7 + j
    in1_index = batch * 7 + j

    # Load and compute
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0) / 8.0
    y_val = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    z_val = tl.load(in1_ptr + in1_index, mask=mask, other=0.0)
    out_val = x_val + y_val + z_val

    # Store
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_add(in0, in1, in2):
    n_elements = in0.numel()
    BLOCK_SIZE = 256
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in0)
    fused_kernel[grid](
        x_ptr=in0,
        in2_ptr=in2,
        in1_ptr=in1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_add