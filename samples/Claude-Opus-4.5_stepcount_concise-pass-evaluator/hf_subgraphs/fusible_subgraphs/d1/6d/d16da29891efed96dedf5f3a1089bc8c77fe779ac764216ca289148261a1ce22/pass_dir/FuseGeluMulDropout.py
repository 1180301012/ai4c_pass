import torch
import triton
import triton.language as tl

# Pattern matching function - matches GELU + multiply + dropout (with training=False)
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for small tensors
@triton.jit
def fused_gelu_mul_kernel_small(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    inv_sqrt_2 = 0.7071067811865476
    x_scaled = x * inv_sqrt_2
    erf_val = tl.math.erf(x_scaled)
    gelu_x = x * 0.5 * (1.0 + erf_val)
    out = gelu_x * y
    
    tl.store(out_ptr + offsets, out, mask=mask)

# Triton kernel for medium/large tensors
@triton.jit
def fused_gelu_mul_kernel_large(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    inv_sqrt_2 = 0.7071067811865476
    x_scaled = x * inv_sqrt_2
    erf_val = tl.math.erf(x_scaled)
    gelu_x = x * 0.5 * (1.0 + erf_val)
    out = gelu_x * y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_gelu_mul(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    
    if n_elements <= 200000:  # Small tensors (131072)
        # Use larger block size for fewer blocks
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_gelu_mul_kernel_small[grid](
            in_0, in_1, out, n_elements,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4,
        )
    elif n_elements <= 5000000:  # Medium tensors (4194304)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_gelu_mul_kernel_large[grid](
            in_0, in_1, out, n_elements,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
        )
    else:  # Large tensors (16777216)
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_gelu_mul_kernel_large[grid](
            in_0, in_1, out, n_elements,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
        )
    
    return out

def replacement_func():
    return fused_gelu_mul