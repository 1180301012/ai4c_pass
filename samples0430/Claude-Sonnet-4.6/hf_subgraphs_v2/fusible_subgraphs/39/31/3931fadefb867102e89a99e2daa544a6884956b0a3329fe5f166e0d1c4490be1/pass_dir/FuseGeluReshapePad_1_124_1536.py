import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _gelu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    gelu_out = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))
    gelu_out = gelu_out.to(x.dtype)
    tl.store(out_ptr + offsets, gelu_out, mask=mask)


@torch.fx.wrap
def triton_gelu_reshape(in_0):
    # Returns [1, 248, 768] (same data as gelu of [1, 124, 1536])
    n_elements = in_0.numel()
    out = torch.empty(1, 248, 768, dtype=in_0.dtype, device=in_0.device)
    BLOCK_SIZE = 2048
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    _gelu_kernel[(num_blocks,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return triton_gelu_reshape