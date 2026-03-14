import torch
import triton
import triton.language as tl


# Pattern matching function - for scalar 0.07216878364870322
def pattern(in_0, in_1):
    """
    Match the computation pattern with scalar 0.07216878364870322:
    1. ReLU activation (inplace)
    2. Flatten starting from dim 2
    3. L2 norm over last dimension (keepdim=True)
    4. Multiply by scalar constant (0.07216878364870322)
    5. Clamp min=1e-05
    6. Divide
    7. Multiply by in_0 (scalar weight)
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_1 = None
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_3 = None
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_4 = None
    tmp_6 = tmp_2 / tmp_5
    tmp_2 = tmp_5 = None
    tmp_7 = tmp_6 * tmp_0
    tmp_6 = tmp_0 = None
    return (tmp_7,)


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel for fused ReLU + Normalize + Multiply (scalar 0.07216878364870322)
@triton.jit
def fused_norm_kernel_072(
    input_ptr,      # in_1 (original input)
    weight_ptr,     # in_0 (scalar weight)
    output_ptr,     # output
    batch_size,     # B
    channels,       # C  
    spatial_size,   # H * W
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate number of programs needed
    num_programs = batch_size * channels
    
    if pid >= num_programs:
        return
    
    # Calculate batch and channel indices
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Calculate base offset for this batch/channel
    base_offset = batch_idx * channels * spatial_size + channel_idx * spatial_size
    
    # Load all data for this batch/channel slice
    # First accumulate sum of squares
    sum_sq = 0.0
    
    # Iterate through blocks if spatial_size > BLOCK_SIZE
    for start in range(0, spatial_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load input and apply ReLU
        x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        x = tl.maximum(x, 0.0)  # ReLU
        
        # Compute sum of squares
        x_sq = x * x
        sum_sq += tl.sum(x_sq, axis=0)
    
    # Compute norm: sqrt(sum_sq)
    norm = tl.sqrt(sum_sq + 1e-12)
    
    # Multiply by norm_factor and clamp
    norm_factor = 0.07216878364870322
    scaled_norm = norm * norm_factor
    scaled_norm = tl.maximum(scaled_norm, 1e-05)
    
    # Load weight
    weight = tl.load(weight_ptr)
    
    # Now compute normalized output
    # We need to reload the data (or we could cache it - let's optimize)
    # Actually, let's recompute in a second pass or store intermediate
    
    # For now, let's do two-pass: first compute norm, then compute output
    # This is still faster than multiple kernel launches
    
    # Second pass: normalize and store
    for start in range(0, spatial_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Load input and apply ReLU
        x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        x = tl.maximum(x, 0.0)  # ReLU
        
        # Normalize and multiply by weight
        normalized = x / scaled_norm
        result = normalized * weight
        
        # Store result
        tl.store(output_ptr + base_offset + offsets, result, mask=mask)


@torch.fx.wrap
def fused_norm_wrapper_072(in_0, in_1):
    """
    Wrapper function that launches the Triton kernel.
    Fuses: ReLU + Flatten + Norm + Mul + Clamp + Div + Mul(weight)
    Scalar: 0.07216878364870322
    """
    # Get input shape: [B, C, H, W]
    B, C, H, W = in_1.shape
    spatial_size = H * W
    
    # Output shape after flatten: [B, C, H*W]
    output_shape = (B, C, spatial_size)
    
    # Allocate output
    output = torch.empty(output_shape, device=in_1.device, dtype=in_1.dtype)
    
    # Calculate grid - each program handles one (batch, channel) pair
    num_programs = B * C
    BLOCK_SIZE = 1024
    
    # Launch kernel
    fused_norm_kernel_072[(num_programs,)](
        in_1,           # input_ptr
        in_0,           # weight_ptr  
        output,         # output_ptr
        B,              # batch_size
        C,              # channels
        spatial_size,   # spatial_size
        BLOCK_SIZE,     # BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return fused_norm_wrapper_072