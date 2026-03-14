import torch
import triton
import triton.language as tl

# Pattern matching function - matches the cat-mul-add chain
def pattern(in_6, in_5, in_2, in_4):
    """
    Match the rotary position embedding computation:
    - Negate in_6
    - Concatenate (-in_6, in_5) along last dimension
    - Multiply by in_2
    - Add to in_4
    - Cast to float32
    """
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    return tmp_4

# Extract arguments from matched nodes
def replacement_args(in_6, in_5, in_2, in_4):
    return (in_6, in_5, in_2, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_cat_mul_add_kernel(
    in_6_ptr, in_5_ptr, in_2_ptr, in_4_ptr, out_ptr,
    n_elements,
    D: tl.constexpr,
    D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: out = in_4 + cat(-in_6, in_5) * in_2
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute position within the last dimension
    d = offsets % D
    row_idx = offsets // D
    
    # Determine if we're in the first half or second half
    is_first_half = d < D_HALF
    
    # Compute index for the half-sized tensors
    half_d = d % D_HALF
    half_idx = row_idx * D_HALF + half_d
    
    # Load values from half-sized tensors
    in_6_val = tl.load(in_6_ptr + half_idx, mask=mask, other=0.0)
    in_5_val = tl.load(in_5_ptr + half_idx, mask=mask, other=0.0)
    
    # Compute the concatenated value: first half is -in_6, second half is in_5
    cat_val = tl.where(is_first_half, -in_6_val, in_5_val)
    
    # Load full-sized tensors
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_4_val = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    
    # Final computation: in_4 + cat_val * in_2
    out = in_4_val + cat_val * in_2_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_cat_mul_add_kernel_broadcast(
    in_6_ptr, in_5_ptr, in_2_ptr, in_4_ptr, out_ptr,
    n_elements,
    S, D,
    D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Variant kernel that handles broadcasting of in_2 from [1, 1, S, D] to [B, H, S, D]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute position within the last dimension
    d = offsets % D
    s = (offsets // D) % S
    row_idx = offsets // D
    
    # Determine if we're in the first half or second half
    is_first_half = d < D_HALF
    
    # Compute index for the half-sized tensors
    half_d = d % D_HALF
    half_idx = row_idx * D_HALF + half_d
    
    # Load values from half-sized tensors
    in_6_val = tl.load(in_6_ptr + half_idx, mask=mask, other=0.0)
    in_5_val = tl.load(in_5_ptr + half_idx, mask=mask, other=0.0)
    
    # Compute the concatenated value
    cat_val = tl.where(is_first_half, -in_6_val, in_5_val)
    
    # For in_2, compute broadcast index: maps to [0, 0, s, d]
    in_2_idx = s * D + d
    in_2_val = tl.load(in_2_ptr + in_2_idx, mask=mask, other=0.0)
    
    # Load in_4 (full shape, no broadcasting)
    in_4_val = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    
    # Final computation
    out = in_4_val + cat_val * in_2_val
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_cat_mul_add(in_6, in_5, in_2, in_4):
    """
    Wrapper function that launches the Triton kernel.
    
    Computes: out = in_4 + torch.cat((-in_6, in_5), dim=-1) * in_2
    """
    # Ensure tensors are contiguous
    in_6 = in_6.contiguous()
    in_5 = in_5.contiguous()
    in_2 = in_2.contiguous()
    in_4 = in_4.contiguous()
    
    D = in_4.shape[-1]
    D_HALF = D // 2
    n_elements = in_4.numel()
    
    # Check if we need broadcasting for in_2
    needs_broadcast = in_2.shape != in_4.shape
    
    # Output tensor with same shape as in_4
    out = torch.empty_like(in_4, dtype=torch.float32)
    
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    if needs_broadcast:
        S = in_4.shape[-2]
        fused_cat_mul_add_kernel_broadcast[grid](
            in_6, in_5, in_2, in_4, out,
            n_elements,
            S, D,
            D_HALF,
        )
    else:
        fused_cat_mul_add_kernel[grid](
            in_6, in_5, in_2, in_4, out,
            n_elements,
            D, D_HALF,
        )
    
    return out


# Replacement function - returns the optimized function
def replacement_func():
    return fused_cat_mul_add