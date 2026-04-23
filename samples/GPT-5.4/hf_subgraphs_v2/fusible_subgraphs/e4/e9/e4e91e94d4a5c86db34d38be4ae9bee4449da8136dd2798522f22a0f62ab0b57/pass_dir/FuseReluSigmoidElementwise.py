import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _relu_sigmoid_small_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    for tile_id in tl.static_range(0, NUM_TILES):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    x = tl.maximum(x, 0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid(in_0):
    n_elements = in_0.numel()
    out = torch.empty(in_0.shape, device=in_0.device, dtype=in_0.dtype)

    if n_elements <= 8192:
        _relu_sigmoid_small_kernel[(1,)](
            in_0,
            out,
            n_elements,
            BLOCK_SIZE=1024,
            NUM_TILES=8,
            num_warps=4,
            num_stages=1,
        )
    else:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _relu_sigmoid_kernel[grid](
            in_0,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=1,
        )

    return out


def replacement_func():
    return fused_relu_sigmoid