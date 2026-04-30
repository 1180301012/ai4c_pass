import torch
import triton
import triton.language as tl


# Pattern: matches in_0.transpose(-2, -1)
def pattern(in_0):
    return in_0.transpose(-2, -1)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    B,
    S,
    D,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose flat output index (layout: B, D, S)
    b = offsets // (D * S)
    d = (offsets // S) % D
    s = offsets % S

    # Corresponding flat input index (layout: B, S, D)
    in_idx = b * (S * D) + s * D + d

    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def triton_transpose(in_0):
    B = in_0.shape[0]
    S = in_0.shape[1]
    D = in_0.shape[2]
    n_elements = B * S * D

    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)  # will hold transposed data, shape [B,S,D]

    transpose_kernel[(num_blocks,)](
        in_0,
        out,
        B, S, D,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return as [B, D, S] — same memory, different logical shape
    return out.view(B, D, S)


def replacement_func():
    return triton_transpose