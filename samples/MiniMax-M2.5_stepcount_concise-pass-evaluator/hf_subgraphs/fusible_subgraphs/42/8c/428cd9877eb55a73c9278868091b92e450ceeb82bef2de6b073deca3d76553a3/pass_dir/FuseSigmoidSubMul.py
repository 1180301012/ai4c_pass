import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_3, in_4, conv_view):
    """
    Match the pattern: cat -> sigmoid -> sub -> mul
    This fuses the activation and arithmetic operations.
    
    Args:
        in_3: First tensor for concatenation [B, 1, 6400]
        in_4: Second tensor for concatenation [B, 1, 1600]
        conv_view: Conv output after view [B, 1, 400]
    
    Returns:
        Final output after sigmoid, sub, mul
    """
    # Concatenate along dimension 2
    tmp_cat = torch.cat([in_3, in_4, conv_view], 2)
    # Apply sigmoid activation
    tmp_sigmoid = tmp_cat.sigmoid()
    # Subtract 0.25
    tmp_sub = tmp_sigmoid - 0.25
    # Multiply by pi
    tmp_mul = tmp_sub * 3.141592653589793
    return tmp_mul


def replacement_args(in_3, in_4, conv_view):
    """
    Extract arguments for the replacement function.
    """
    return (in_3, in_4, conv_view)


# Optimized Triton kernel with fast math
@triton.jit
def fused_sigmoid_sub_mul_kernel(
    in_3_ptr,
    in_4_ptr,
    conv_view_ptr,
    out_ptr,
    size_0,
    size_1,
    size_2,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Load from three tensors (simulating concatenation)
    2. Compute: (sigmoid(x) - 0.25) * pi using fast math
    
    Uses a flat 1D grid where each program handles a contiguous block.
    Uses tl.math.exp for faster exponential computation.
    """
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    start_offset = pid * BLOCK_SIZE
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate batch index and position within batch
    batch_size = size_0 + size_1 + size_2
    batch_idx = offsets // batch_size
    pos_in_batch = offsets % batch_size
    
    # Create masks for each tensor region
    mask_3 = pos_in_batch < size_0
    mask_4 = (pos_in_batch >= size_0) & (pos_in_batch < size_0 + size_1)
    mask_conv = pos_in_batch >= (size_0 + size_1)
    
    # Compute offsets within each tensor
    offset_3 = batch_idx * size_0 + pos_in_batch
    offset_4 = batch_idx * size_1 + (pos_in_batch - size_0)
    offset_conv = batch_idx * size_2 + (pos_in_batch - size_0 - size_1)
    
    # Load from each tensor with appropriate mask
    x_3 = tl.load(in_3_ptr + offset_3, mask=mask & mask_3, other=0.0)
    x_4 = tl.load(in_4_ptr + offset_4, mask=mask & mask_4, other=0.0)
    x_conv = tl.load(conv_view_ptr + offset_conv, mask=mask & mask_conv, other=0.0)
    
    # Select the appropriate value based on position
    x = tl.where(mask_3, x_3, tl.where(mask_4, x_4, x_conv))
    
    # Compute: (sigmoid(x) - 0.25) * pi
    # Precompute constants
    pi = 3.141592653589793
    offset = 0.25 * pi  # Precompute offset
    
    # Use fast sigmoid approximation: 1 / (1 + exp(-x))
    # Use tl.math.exp for potentially faster computation
    neg_x = -x
    exp_neg_x = tl.math.exp(neg_x)
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    
    # Compute result with precomputed offset: sigmoid * pi - 0.25 * pi
    result = sigmoid_x * pi - offset
    
    # Compute output offset
    out_offset = batch_idx * batch_size + pos_in_batch
    
    # Store result
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_sub_mul(in_3, in_4, conv_view):
    """
    Wrapper function to launch the fused kernel.
    
    Args:
        in_3: Tensor of shape [B, 1, 6400]
        in_4: Tensor of shape [B, 1, 1600]
        conv_view: Tensor of shape [B, 1, 400]
    
    Returns:
        Tensor of shape [B, 1, 8400] after fused sigmoid-sub-mul
    """
    # Get batch size and reshape to 2D
    B = in_3.shape[0]
    in_3_flat = in_3.view(B, -1)  # [B, 6400]
    in_4_flat = in_4.view(B, -1)  # [B, 1600]
    conv_view_flat = conv_view.view(B, -1) # [B, 400]
    
    # Get sizes
    size_0 = in_3_flat.shape[1]  # 6400
    size_1 = in_4_flat.shape[1]  # 1600
    size_2 = conv_view_flat.shape[1]  # 400
    total_elements = B * (size_0 + size_1 + size_2)
    
    # Create output tensor [B, total_elements_per_batch]
    batch_elements = size_0 + size_1 + size_2
    out = torch.empty((B, batch_elements), dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel with 1D grid - use optimal block size for this workload
    BLOCK_SIZE = 512
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    fused_sigmoid_sub_mul_kernel[grid](
        in_3_ptr=in_3_flat,
        in_4_ptr=in_4_flat,
        conv_view_ptr=conv_view_flat,
        out_ptr=out,
        size_0=size_0,
        size_1=size_1,
        size_2=size_2,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to [B, 1, 8400]
    return out.view(B, 1, -1)


def replacement_func():
    return fused_sigmoid_sub_mul