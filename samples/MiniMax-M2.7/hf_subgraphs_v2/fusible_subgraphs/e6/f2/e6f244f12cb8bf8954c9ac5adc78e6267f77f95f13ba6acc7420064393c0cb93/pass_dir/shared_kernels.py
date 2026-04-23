"""
Shared Triton Kernels for Fused Add + Mean Optimization

This module contains the shared kernel implementations that are used by all pass files.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Triton Kernels for Fused Add + Mean
# ============================================================================

@triton.jit
def fused_add_mean_2inputs_kernel(
    in_ptr_a,
    in_ptr_b,
    out_sum_ptr,
    out_mean_ptr,
    n_elements,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for 2-input addition and mean reduction."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(in_ptr_a + offsets, mask=mask, other=0.0)
    b = tl.load(in_ptr_b + offsets, mask=mask, other=0.0)
    
    sum_val = a + b
    tl.store(out_sum_ptr + offsets, sum_val, mask=mask)
    
    mean_val = sum_val / spatial_size
    tl.store(out_mean_ptr + offsets, mean_val, mask=mask)


@triton.jit
def fused_add_mean_3inputs_kernel(
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    out_sum_ptr,
    out_mean_ptr,
    n_elements,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for 3-input addition and mean reduction."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(in_ptr_a + offsets, mask=mask, other=0.0)
    b = tl.load(in_ptr_b + offsets, mask=mask, other=0.0)
    c = tl.load(in_ptr_c + offsets, mask=mask, other=0.0)
    
    sum_val = a + b + c
    tl.store(out_sum_ptr + offsets, sum_val, mask=mask)
    
    mean_val = sum_val / spatial_size
    tl.store(out_mean_ptr + offsets, mean_val, mask=mask)


@triton.jit
def fused_add_mean_1input_kernel(
    in_ptr,
    out_sum_ptr,
    out_mean_ptr,
    n_elements,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for 1-input (identity) and mean reduction."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    val = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_sum_ptr + offsets, val, mask=mask)
    
    mean_val = val / spatial_size
    tl.store(out_mean_ptr + offsets, mean_val, mask=mask)


# ============================================================================
# Kernel Wrappers
# ============================================================================

def fused_add_mean_2inputs_impl(in_a, in_b):
    """Implementation for 2-input fused add + mean kernel."""
    B, C, H, W = in_a.shape
    n_elements = B * C * H * W
    spatial_size = H * W
    
    out_sum = torch.empty_like(in_a)
    out_mean = torch.empty_like(in_a)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_mean_2inputs_kernel[(num_programs,)](
        in_a, in_b, out_sum, out_mean, n_elements, spatial_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sum, out_mean


def fused_add_mean_3inputs_impl(in_a, in_b, in_c):
    """Implementation for 3-input fused add + mean kernel."""
    B, C, H, W = in_a.shape
    n_elements = B * C * H * W
    spatial_size = H * W
    
    out_sum = torch.empty_like(in_a)
    out_mean = torch.empty_like(in_a)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_mean_3inputs_kernel[(num_programs,)](
        in_a, in_b, in_c, out_sum, out_mean, n_elements, spatial_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sum, out_mean


def fused_add_mean_1input_impl(in_tensor):
    """Implementation for 1-input (identity) fused add + mean kernel."""
    B, C, H, W = in_tensor.shape
    n_elements = B * C * H * W
    spatial_size = H * W
    
    out_sum = torch.empty_like(in_tensor)
    out_mean = torch.empty_like(in_tensor)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_mean_1input_kernel[(num_programs,)](
        in_tensor, out_sum, out_mean, n_elements, spatial_size, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_sum, out_mean


# ============================================================================
# Unified Dispatch Wrapper (ALL passes use this)
# ============================================================================

@torch.fx.wrap
def fused_add_mean_dispatch(*args):
    """Unified dispatch wrapper that routes to the appropriate kernel."""
    route = args[-1]
    
    if route == "2input_01":
        return fused_add_mean_2inputs_impl(args[0], args[1])
    elif route == "2input_10":
        return fused_add_mean_2inputs_impl(args[1], args[0])
    elif route == "3input_01":
        return fused_add_mean_3inputs_impl(args[0], args[1], args[2])
    elif route == "3input_10":
        return fused_add_mean_3inputs_impl(args[2], args[0], args[1])
    elif route == "1input":
        return fused_add_mean_1input_impl(args[0])
    else:
        raise ValueError(f"Unknown route: {route}")