import torch
import triton
import triton.language as tl


@triton.jit
def arange_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, offsets, mask=mask)


@triton.jit
def cast_to_bool_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = x != 0
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_arange_cast_dispatch(in_0, route):
    if route == "arange_128":
        end = 128
    elif route == "arange_256":
        end = 256
    elif route == "arange_512":
        end = 512
    elif route == "arange_1024":
        end = 1024
    else:
        raise ValueError(f"Unknown route: {route}")

    # Generate arange
    arange_out = torch.empty(end, dtype=torch.int64, device=torch.device(type='cuda', index=0))
    BLOCK_SIZE_ARANGE = 1024
    n_arange = end
    grid_arange = (triton.cdiv(n_arange, BLOCK_SIZE_ARANGE),)
    arange_kernel[grid_arange](arange_out, n_arange, BLOCK_SIZE=BLOCK_SIZE_ARANGE)

    # Cast to bool
    bool_out = torch.empty_like(in_0, dtype=torch.bool)
    n_cast = in_0.numel()
    BLOCK_SIZE_CAST = 1024
    grid_cast = (triton.cdiv(n_cast, BLOCK_SIZE_CAST),)
    cast_to_bool_kernel[grid_cast](in_0, bool_out, n_cast, BLOCK_SIZE=BLOCK_SIZE_CAST)

    return (arange_out, bool_out)