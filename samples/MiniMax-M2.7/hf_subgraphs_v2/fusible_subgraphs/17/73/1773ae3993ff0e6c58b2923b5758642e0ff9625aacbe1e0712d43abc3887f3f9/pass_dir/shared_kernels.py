"""Shared kernels for cat and layer_norm optimizations"""
import torch
import triton
import triton.language as tl


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def triton_cat_kernel(
    in_2_ptr, in_5_ptr, in_3_ptr,
    out_ptr,
    B0: tl.constexpr, H0: tl.constexpr, W0: tl.constexpr, C: tl.constexpr,
    W1: tl.constexpr, W2: tl.constexpr,
    TOTAL_ELEMENTS: tl.constexpr,  # Total output elements = B0 * H0 * (W0+W1+W2) * C
    BLOCK_SIZE: tl.constexpr
):
    """Optimized cat kernel along dim=2"""
    pid = tl.program_id(0)
    
    # Each thread handles BLOCK_SIZE consecutive output elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL_ELEMENTS
    
    # Decode the output position (flattened)
    # Output shape: [B0, H0, W0+W1+W2, C]
    c = offsets % C
    rem = offsets // C
    w_out = rem % (W0 + W1 + W2)
    rem = rem // (W0 + W1 + W2)
    h = rem % H0
    b = rem // H0
    
    # Determine which input tensor each element comes from
    # in_2 covers [0, W0), in_5 covers [W0, W0+W1), in_3 covers [W0+W1, W0+W1+W2)
    
    # Calculate base offsets for each tensor
    # For contiguous tensors [B, H, W, C], offset = ((b*H + h)*W + w)*C + c
    base_2 = b * H0 * W0 * C + h * W0 * C
    base_5 = b * H0 * W1 * C + h * W1 * C
    base_3 = b * H0 * W2 * C + h * W2 * C
    
    # Compute load offsets for each tensor
    offs_2 = base_2 + w_out * C + c
    offs_5 = base_5 + (w_out - W0) * C + c
    offs_3 = base_3 + (w_out - W0 - W1) * C + c
    
    # Masks for each input region
    mask_0 = (w_out < W0) & (W0 > 0)
    mask_1 = (w_out >= W0) & (w_out < W0 + W1) & (W1 > 0)
    mask_2 = (w_out >= W0 + W1) & (W2 > 0)
    
    # Load values with appropriate masking
    if W0 > 0:
        val = tl.load(in_2_ptr + offs_2, mask=mask & mask_0[:, None], other=0.0)
    
    if W1 > 0:
        val1 = tl.load(in_5_ptr + offs_5, mask=mask & mask_1[:, None], other=0.0)
        if W0 > 0:
            val = tl.where(mask_0[:, None], val, val1)
        else:
            val = val1
    
    if W2 > 0:
        val2 = tl.load(in_3_ptr + offs_3, mask=mask & mask_2[:, None], other=0.0)
        if W0 > 0 or W1 > 0:
            val = tl.where(mask_0[:, None], val, 
                    tl.where(mask_1[:, None], val, val2))
        else:
            val = val2
    
    # Store to output
    store_ptr = out_ptr + offsets
    tl.store(store_ptr, val, mask=mask)


@triton.jit
def triton_layer_norm_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    N: tl.constexpr,  # normalized dimension (C) - may not be power of 2
    M: tl.constexpr,  # total number of elements / C (rows)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr  # Must be power of 2 >= N
):
    """Optimized layer_norm kernel - handles non-power-of-2 N"""
    pid = tl.program_id(0)
    
    # Each program handles one row of M
    row_id = pid
    row_mask = row_id < M
    
    # Load row of data with padding handling
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load the row
    x = tl.load(in_ptr + row_id * N + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean using only valid elements
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance using only valid elements
    x_centered = x - mean
    x_centered_sq = x_centered * x_centered
    var = tl.sum(x_centered_sq, axis=0) / N
    
    # Normalize
    x_norm = x_centered / tl.sqrt(var + eps)
    
    # Load weight and bias (using mask for valid elements only)
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Apply affine transform
    output = x_norm * w + b
    
    # Store with mask for valid elements only
    tl.store(out_ptr + row_id * N + offs, output, mask=mask)


# ============================================================================
# Implementation functions
# ============================================================================

def _cat_impl(in_2, in_5, in_3):
    """Optimized cat implementation"""
    B, H, W0, C = in_2.shape
    _, _, W1, _ = in_5.shape
    _, _, W2, _ = in_3.shape
    W_out = W0 + W1 + W2
    
    out = torch.empty((B, H, W_out, C), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_SIZE = 128
    total_elements = B * H * W_out * C
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_programs,)
    
    triton_cat_kernel[grid](
        in_2, in_5, in_3,
        out,
        B, H, W0, C,
        W1, W2,
        total_elements,  # TOTAL_ELEMENTS
        BLOCK_SIZE
    )
    
    return out


def _next_power_of_2(n):
    """Calculate next power of 2 >= n"""
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _layer_norm_impl(in_4, in_1, in_0):
    """Optimized layer_norm implementation"""
    C = in_4.shape[-1]
    M = in_4.numel() // C
    
    out = torch.empty_like(in_4)
    
    # Use next power of 2 for BLOCK_SIZE (required by tl.arange)
    BLOCK_SIZE = _next_power_of_2(C)
    num_programs = M
    
    triton_layer_norm_kernel[(num_programs,)](
        in_4, in_1, in_0, out,
        C, M,
        1e-12,
        BLOCK_SIZE
    )
    
    return out


# ============================================================================
# Dynamic Dispatch Function (used by both cat and layer_norm passes)
# ============================================================================
def dynamic_dispatch(*args):
    """Routes to the appropriate implementation based on the last argument (route)"""
    if len(args) == 4 and args[-1] == "cat":
        return _cat_impl(args[0], args[1], args[2])
    elif len(args) == 4 and args[-1] == "layer_norm":
        return _layer_norm_impl(args[0], args[1], args[2])
    else:
        raise ValueError(f"Unknown dispatch signature: {args}")