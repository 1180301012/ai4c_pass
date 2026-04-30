import torch
import triton
import triton.language as tl


@triton.jit
def _add_flat_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_reshape_layernorm(x, y):
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    out = torch.empty((rows, hidden_size), device=x.device, dtype=x.dtype)
    n_elements = x.numel()

    if n_elements <= 256:
        block_size = 256
        num_warps = 1
    elif n_elements <= 1024:
        block_size = 512
        num_warps = 2
    elif n_elements <= 4096:
        block_size = 1024
        num_warps = 4
    else:
        block_size = 2048
        num_warps = 8

    grid = (triton.cdiv(n_elements, block_size),)
    _add_flat_kernel[grid](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out