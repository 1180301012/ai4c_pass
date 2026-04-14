import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def relu_dropout_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU (max(x,0)) and store to flattened output
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


@torch.fx.wrap
def relu_dropout_flatten_fused(in_0):
    B = in_0.shape[0]
    n_elements = in_0.numel()
    flat_size = n_elements // B
    out = torch.empty((B, flat_size), dtype=in_0.dtype, device=in_0.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_dropout_flatten_kernel[grid](in_0, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return relu_dropout_flatten_fused