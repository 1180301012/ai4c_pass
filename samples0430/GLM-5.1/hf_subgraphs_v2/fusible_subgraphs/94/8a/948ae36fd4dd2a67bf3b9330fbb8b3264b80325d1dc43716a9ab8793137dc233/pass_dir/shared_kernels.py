import torch
import triton
import triton.language as tl
from torch import device

# Triton kernels - simple fixed config for small tensors
@triton.jit
def cast_int64_to_bool_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    bool_vals = input_vals != 0
    tl.store(output_ptr + offsets, bool_vals, mask=mask)

@triton.jit
def arange_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = offsets.to(tl.int64)
    tl.store(output_ptr + offsets, values, mask=mask)

# Fused kernel - does both arange and cast in one launch
@triton.jit
def fused_arange_cast_kernel(
    arange_ptr,
    cast_ptr,
    input_ptr,
    arange_n,
    cast_n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Cast: int64 -> bool
    cast_mask = offsets < cast_n_elements
    input_vals = tl.load(input_ptr + offsets, mask=cast_mask, other=0)
    bool_vals = input_vals != 0
    tl.store(cast_ptr + offsets, bool_vals, mask=cast_mask)
    
    # Arange: sequential integers (only for offsets < arange_n)
    arange_mask = offsets < arange_n
    arange_vals = offsets.to(tl.int64)
    tl.store(arange_ptr + offsets, arange_vals, mask=arange_mask)

# Kernel wrappers - these are wrapped so the tracer treats them as leaf nodes
@torch.fx.wrap
def triton_cast_int64_to_bool(input_tensor):
    n_elements = input_tensor.numel()
    output = torch.empty(input_tensor.shape, dtype=torch.bool, device=input_tensor.device)
    # Use larger block size for small tensors to reduce scheduling overhead
    BLOCK_SIZE = 2048 if n_elements <= 2048 else 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    cast_int64_to_bool_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

@torch.fx.wrap
def triton_arange(n):
    output = torch.empty(n, dtype=torch.int64, device='cuda')
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    arange_kernel[grid](
        output_ptr=output,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

@torch.fx.wrap
def triton_fused_arange_cast(input_tensor, arange_n):
    cast_n_elements = input_tensor.numel()
    arange_output = torch.empty(arange_n, dtype=torch.int64, device=input_tensor.device)
    cast_output = torch.empty(input_tensor.shape, dtype=torch.bool, device=input_tensor.device)
    BLOCK_SIZE = 1024
    grid = ((cast_n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_arange_cast_kernel[grid](
        arange_ptr=arange_output,
        cast_ptr=cast_output,
        input_ptr=input_tensor,
        arange_n=arange_n,
        cast_n_elements=cast_n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (arange_output, cast_output)

# Dispatch wrapper - NOT wrapped so tracer can see individual outputs
# This function object must be shared across all passes to satisfy output_pass_replacement_func_limit
def dispatch_wrapper(*args):
    route = args[-1]
    args = args[:-1]
    if route == "cast_int64_to_bool":
        return triton_cast_int64_to_bool(args[0])
    elif route == "fuse_arange_128_cast_bool":
        return triton_fused_arange_cast(args[0], 128)
    elif route == "fuse_arange_256_cast_bool":
        return triton_fused_arange_cast(args[0], 256)
    elif route == "fuse_arange_512_cast_bool":
        return triton_fused_arange_cast(args[0], 512)
    elif route == "fuse_arange_1024_cast_bool":
        return triton_fused_arange_cast(args[0], 1024)
    else:
        raise ValueError(f"Unknown route: {route}")