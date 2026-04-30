import torch
import triton
import triton.language as tl


@triton.jit
def flatten_copy_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Contiguous copy: reads x[B, C, 1, 1] flat and writes to out[B, C].
    Since dropout(p=0) is identity and flatten is a view of a contiguous
    tensor, a direct flat copy is semantically correct.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_relu_flatten(in_0):
    """
    Replaces: dropout(p=0, training=False) -> flatten(1, -1)
    Input: [B, C, 1, 1] already relu'd contiguous tensor.
    Output: [B, C] contiguous tensor.
    Uses a Triton copy kernel (dropout is identity, flatten is a view of contiguous tensor).
    """
    B = in_0.shape[0]
    C = in_0.shape[1]
    n = B * C  # = in_0.numel()

    # torch.empty: only whitelisted factory op available for output allocation
    out = torch.empty((B, C), dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 2048
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    flatten_copy_kernel[(n_blocks,)](in_0, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def pattern(in_0):
    """
    Match: dropout -> flatten(1, -1)
    The graph has relu -> dropout -> flatten.
    Dropout(p=0, training=False) is identity; flatten is a view.
    """
    tmp_1 = torch.nn.functional.dropout(in_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_relu_flatten