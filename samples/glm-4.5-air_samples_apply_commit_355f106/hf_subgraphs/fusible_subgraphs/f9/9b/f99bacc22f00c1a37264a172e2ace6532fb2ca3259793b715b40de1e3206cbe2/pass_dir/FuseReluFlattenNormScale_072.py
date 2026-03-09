import torch
import triton
import triton.language as tl


# Pattern matching function - matches the entire computation graph with different constant
def pattern(in_0, in_1):
    # Step 1: ReLU activation
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    
    # Step 2: Flatten spatial dimensions (start_dim=2)
    tmp_2 = torch.flatten(tmp_1, 2)
    
    # Step 3: L2 normalization over last dimension
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    
    # Step 4: Multiply by constant (scale factor 0.07216878364870322)
    tmp_4 = tmp_3 * 0.07216878364870322
    
    # Step 5: Clamp minimum value (for numerical stability)
    tmp_5 = tmp_4.clamp(min=1e-05)
    
    # Step 6: Divide normalized tensor by clamped norm
    tmp_6 = tmp_2 / tmp_5
    
    # Step 7: Multiply by scalar weight
    tmp_7 = tmp_6 * in_0
    
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel that fuses all operations (same as the other pass)
@triton.jit
def relu_flatten_norm_scale_kernel_072(
    in_ptr,        # input tensor (in_1)
    weight_ptr,    # scalar weight (in_0)
    out_ptr,       # output tensor
    scale_factor: tl.constexpr,  # constant scale factor
    B: tl.constexpr,   # batch size
    C: tl.constexpr,   # channels
    H: tl.constexpr,   # height
    W: tl.constexpr,   # width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one channel within one batch element
    # Compute the flattened spatial size
    spatial_size = H * W
    
    # Get program id
    pid = tl.program_id(0)
    
    # Compute batch and channel indices
    num_channels = B * C
    if pid >= num_channels:
        return
    
    batch_idx = pid // C
    channel_idx = pid % C
    
    # Compute offsets for this channel
    # Input layout: [B, C, H, W]
    base_offset = batch_idx * C * H * W + channel_idx * H * W
    
    # Load the scalar weight
    weight = tl.load(weight_ptr)
    
    # Compute L2 norm first (need to load all values for this channel)
    # Use BLOCK_SIZE for vectorized loading
    norm_sq = 0.0
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load values for this channel
        ptrs = base_offset + offsets
        x = tl.load(in_ptr + ptrs, mask=mask, other=0.0)
        
        # Apply ReLU (x > 0 ? x : 0)
        x = tl.where(x > 0, x, 0.0)
        
        # Accumulate squared values for norm
        norm_sq += tl.sum(x * x, mask=mask)
    
    # Compute L2 norm
    norm = tl.sqrt(norm_sq + 1e-8)
    
    # Apply scale and clamp
    scaled_norm = norm * scale_factor
    clamped_norm = tl.where(scaled_norm > 1e-05, scaled_norm, 1e-05)
    
    # Second pass: normalize and store
    for i in range(0, spatial_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load values
        ptrs = base_offset + offsets
        x = tl.load(in_ptr + ptrs, mask=mask, other=0.0)
        
        # Apply ReLU
        x = tl.where(x > 0, x, 0.0)
        
        # Normalize
        normalized = x / clamped_norm
        
        # Scale by weight
        result = normalized * weight
        
        # Store result
        tl.store(out_ptr + ptrs, result, mask=mask)


@torch.fx.wrap
def relu_flatten_norm_scale_wrapper_072(in_0, in_1):
    """
    Fused kernel: ReLU -> Flatten -> L2 Norm -> Scale -> Clamp -> Divide -> Multiply
    All in a single GPU kernel launch.
    """
    # Get input shape
    B, C, H, W = in_1.shape
    
    # Scale factor for this pattern
    scale_factor = 0.07216878364870322
    
    # Allocate output with same shape as input
    out = torch.empty_like(in_1)
    
    # Calculate grid size
    # Each program handles one channel of one batch element
    num_programs = B * C
    
    # BLOCK_SIZE for each program
    BLOCK_SIZE = 128
    
    # Launch kernel
    relu_flatten_norm_scale_kernel_072[(num_programs,)](
        in_ptr=in_1,
        weight_ptr=in_0,
        out_ptr=out,
        scale_factor=scale_factor,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return relu_flatten_norm_scale_wrapper_072