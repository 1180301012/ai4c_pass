import torch
import triton
import triton.language as tl


def pattern(x):
    tmp = x.to(dtype=torch.float32)
    tmp = 1.0 - tmp
    tmp = tmp * -3.4028234663852886e+38
    return tmp


def replacement_args(x):
    return (x,)


@triton.jit
def attention_mask_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_f32 = x.to(tl.float32)
    result = (1.0 - x_f32) * -3.4028234663852886e+38

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_attention_mask(x):
    n_elements = x.numel()
    # Output is always float32 (from the .to(dtype=torch.float32) + arithmetic)
    out = torch.empty(n_elements, dtype=torch.float32, device=x.device)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    attention_mask_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape to match input shape
    if x.dim() == 4:
        return out.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    elif x.dim() == 3:
        return out.view(x.shape[0], x.shape[1], x.shape[2])
    elif x.dim() == 2:
        return out.view(x.shape[0], x.shape[1])
    else:
        return out.view(x.shape[0])


def replacement_func():
    return fused_attention_mask