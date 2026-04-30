import torch
import triton
import triton.language as tl


def pattern(tmp_9, in_9, tmp_12, tmp_13):
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17


def replacement_args(tmp_9, in_9, tmp_12, tmp_13):
    return (tmp_9, in_9, tmp_12, tmp_13)


@triton.jit
def fused_sigmoid_gate_kernel(
    tmp_9_ptr,
    in_9_ptr,
    tmp_12_ptr,
    tmp_13_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load all inputs
    a = tl.load(tmp_9_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(in_9_ptr + offsets, mask=mask).to(tl.float32)
    c = tl.load(tmp_12_ptr + offsets, mask=mask).to(tl.float32)
    d = tl.load(tmp_13_ptr + offsets, mask=mask).to(tl.float32)

    # sigmoid(tmp_9) * tmp_12 + sigmoid(in_9) * tmp_13
    result = tl.sigmoid(a) * c + tl.sigmoid(b) * d

    # Store
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_gate(tmp_9, in_9, tmp_12, tmp_13):
    # tmp_9: [300, 1, 256], in_9: [300, 1, 256]
    # tmp_12: [300, 256], tmp_13: [300, 1, 256]
    # output: [300, 1, 256]
    output = torch.empty_like(tmp_9)
    n_elements = tmp_9.numel()

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_sigmoid_gate_kernel[grid](
        tmp_9, in_9, tmp_12, tmp_13, output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return output


def replacement_func():
    return fused_sigmoid_gate