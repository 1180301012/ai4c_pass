import torch
import triton
import triton.language as tl

# Pattern matching function - matches hardtanh operation
def pattern(x):
    """
    Match hardtanh(x, 0.0, 6.0, False) which is essentially ReLU6
    """
    result = torch.nn.functional.hardtanh(x, 0.0, 6.0, False)
    return result

# Argument extraction function
def replacement_args(x):
    return (x,)

# Persistent kernel for small tensors
@triton.jit
def hardtanh_persistent_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_chunks = tl.cdiv(n_elements, BLOCK_SIZE)
    
    for chunk_idx in range(pid, num_chunks, NUM_BLOCKS):
        block_start = chunk_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        out = tl.minimum(tl.maximum(x, 0.0), 6.0)
        tl.store(out_ptr + offsets, out, mask=mask)

# Simple kernel for large tensors
@triton.jit
def hardtanh_simple_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def triton_hardtanh(x):
    out = torch.empty_like(x)
    n = x.numel()
    
    if n < 500000:
        # Small tensor: persistent kernel - best config
        hardtanh_persistent_kernel[(16,)](
            x, out, n,
            NUM_BLOCKS=16,
            BLOCK_SIZE=4096,
            num_warps=4,
        )
    else:
        # Large tensor: parallel kernel
        hardtanh_simple_kernel[(triton.cdiv(n, 4096),)](
            x, out, n,
            BLOCK_SIZE=4096,
            num_warps=4,
        )
    
    return out

# Replacement function
def replacement_func():
    return triton_hardtanh