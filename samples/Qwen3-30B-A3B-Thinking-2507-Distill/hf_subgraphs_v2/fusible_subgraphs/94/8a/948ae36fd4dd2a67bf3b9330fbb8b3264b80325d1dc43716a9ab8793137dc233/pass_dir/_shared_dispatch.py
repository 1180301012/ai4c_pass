import torch
import triton
import triton.language as tl


@triton.jit
def _shared_bool_kernel_128(
    in_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x != 0, mask=mask)


@triton.jit
def _shared_bool_kernel_256(
    in_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x != 0, mask=mask)


@triton.jit
def _shared_bool_kernel_512(
    in_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x != 0, mask=mask)


@triton.jit
def _shared_bool_kernel_1024(
    in_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x != 0, mask=mask)


@torch.fx.wrap
def dispatch_bool(in_0, route):
    # Output shape must match arange's shape: [N] (not [B, N])
    # The bool conversion of in_0 produces shape [N] matching the arange output
    N = in_0.shape[-1]  # sequence length N (e.g., 128, 256, 512, 1024)
    n = in_0.numel()    # total elements B * N
    out = torch.empty(N, dtype=torch.bool, device=in_0.device)
    BLOCK_SIZE = 128
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    if route == "128":
        _shared_bool_kernel_128[grid](in_0, out, n, BLOCK_SIZE=BLOCK_SIZE)
    elif route == "256":
        _shared_bool_kernel_256[grid](in_0, out, n, BLOCK_SIZE=BLOCK_SIZE)
    elif route == "512":
        _shared_bool_kernel_512[grid](in_0, out, n, BLOCK_SIZE=BLOCK_SIZE)
    elif route == "1024":
        _shared_bool_kernel_1024[grid](in_0, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out