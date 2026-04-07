import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """Match transpose operation swapping last two dimensions"""
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized transpose kernel
@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    orig_stride_1: tl.constexpr,
    orig_stride_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance kernel for transposing last two dimensions"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load linear index from output position
    output_idx = tl.load(out_ptr + offsets, mask=mask)
    
    # Convert linear index to original coordinates
    # For shape [B, C, H, W] -> [B, C, W, H], the mapping is:
    # output_idx = b * (C * W * H) + c * (W * H) + w * H + h
    # We need to reverse this to get input coordinates:
    h_orig = output_idx % orig_stride_2
    w_orig = (output_idx // orig_stride_2) % orig_stride_1
    bc_orig = output_idx // (orig_stride_1 * orig_stride_2)
    b = bc_orig // orig_stride_2
    c = bc_orig % orig_stride_2
    
    # Calculate input linear index
    input_idx = b * (orig_stride_1 * orig_stride_2) + c * (orig_stride_1 * orig_stride_2) + h_orig * orig_stride_1 + w_orig
    
    # Load data from original position
    x = tl.load(x_ptr + input_idx, mask=tl.arange(0, BLOCK_SIZE) < (n_elements - block_start), other=0.0)
    
    # Store at output position (which is already the correct location)
    tl.store(out_ptr + offsets, x, mask=mask)

# Alternative approach using stride-based indexing
@triton.jit
def transpose_kernel_stride_optimized(
    x_ptr,
    out_ptr,
    n_elements,
    orig_shape_0: tl.constexpr,
    orig_shape_1: tl.constexpr,
    orig_shape_2: tl.constexpr,
    orig_shape_3: tl.constexpr,
    orig_stride_0: tl.constexpr,
    orig_stride_1: tl.constexpr,
    orig_stride_2: tl.constexpr,
    orig_stride_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """More efficient transpose kernel using proper stride mapping"""
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    if tl.all(~mask):
        return
    
    # Calculate linear strides for transposed tensor [B, C, W, H]
    trans_stride_0 = orig_shape_1 * orig_shape_3 * orig_shape_2  # C * W * H
    trans_stride_1 = orig_shape_3 * orig_shape_2                 # W * H
    trans_stride_2 = orig_shape_2                               # H
    trans_stride_3 = 1                                          # 1
    
    # Convert linear output index to transposed tensor coordinates [b, c, w, h]
    # offsets gives us the linear index in the output
    b = offsets // trans_stride_0
    remainder = offsets % trans_stride_0
    c = remainder // trans_stride_1
    remainder = remainder % trans_stride_1
    w = remainder // trans_stride_2
    h = remainder % trans_stride_2
    
    # Convert original coordinates [b, c, h, w] to linear input index
    input_idx = b * orig_stride_0 + c * orig_stride_1 + h * orig_stride_2 + w * orig_stride_3
    
    # Load from original position and store to output position
    x = tl.load(x_ptr + input_idx, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_transpose_last_dims(x):
    """Wrapper function for transpose optimization using Triton kernel"""
    # Get tensor properties for 4D tensor [B, C, H, W]
    orig_shape = x.shape
    B, C, H, W = orig_shape
    
    # Calculate strides (we'll assume contiguous memory for now)
    # For performance analysis, we'll use the stride-based approach
    stride_B = C * H * W
    stride_C = H * W
    stride_H = W
    stride_W = 1
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Use optimized block size based on tensor size
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with transposed shape [B, C, W, H]
    out = torch.empty((B, C, W, H), dtype=x.dtype, device=x.device)
    
    # Launch optimized Triton kernel
    transpose_kernel_stride_optimized[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        orig_shape_0=B,
        orig_shape_1=C, 
        orig_shape_2=H,
        orig_shape_3=W,
        orig_stride_0=stride_B,
        orig_stride_1=stride_C,
        orig_stride_2=stride_H,
        orig_stride_3=stride_W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return triton_transpose_last_dims