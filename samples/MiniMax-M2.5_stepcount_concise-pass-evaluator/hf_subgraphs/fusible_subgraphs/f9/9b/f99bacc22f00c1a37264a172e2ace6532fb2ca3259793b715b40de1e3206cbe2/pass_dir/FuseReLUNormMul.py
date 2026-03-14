import torch
import triton
import triton.language as tl


# Pattern matching function - must exactly match the computation in model.py
def pattern(in_0, in_1):
    """
    Match the computation pattern:
    1. ReLU activation (inplace)
    2. Flatten starting from dim 2
    3. L2 norm over last dimension (keepdim=True)
    4. Multiply by scalar constant
    5. Clamp min=1e-05
    6. Divide
    7. Multiply by in_0 (scalar weight)
    """
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_1 = None
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.14433756729740643
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


# Optimized Triton kernel for fused ReLU + Normalize + Multiply
@triton.jit
def fused_norm_kernel(
    input_ptr,      # in_1 (original input)
    weight_ptr,     # in_0 (scalar weight)
    output_ptr,     # output
    norm_factor_ptr, # scalar constant (0.14433756729740643)
    batch_size,     # B
    channels,       # C  
    spatial_size,   # H * W
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate number of programs needed
    num_programs = batch_size * channels
    num_pid = pid
    
    if num_pid >= num_programs:
        return
    
    # Calculate batch and channel indices
    batch_idx = num_pid // channels
    channel_idx = num_pid % channels
    
    # Calculate offsets
    offset = batch_idx * channels * spatial_size + channel_idx * spatial_size
    
    # Load input data for this batch/channel
    # Each program processes one (batch, channel) slice of size spatial_size
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size
    
    # Load input and apply ReLU
    x = tl.load(input_ptr + offset + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)  # ReLU
    
    # Compute sum of squares for L2 norm
    x_sq = x * x
    
    # Reduce sum across spatial dimension
    # For BLOCK_SIZE < spatial_size, we need multiple loads
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute norm: sqrt(sum + EPS)
    # Note: original has clamp after multiply by scalar, we incorporate that
    norm = tl.sqrt(sum_sq + 1e-12)
    
    # Multiply by norm_factor and clamp
    norm_factor = 0.14433756729740643
    scaled_norm = norm * norm_factor
    scaled_norm = tl.maximum(scaled_norm, 1e-05)
    
    # Normalize and multiply by weight
    weight = tl.load(weight_ptr)
    normalized = x / scaled_norm
    result = normalized * weight
    
    # Store result
    tl.store(output_ptr + offset + offsets, result, mask=mask)


@torch.fx.wrap
def fused_norm_wrapper(in_0, in_1):
    """
    Wrapper function that launches the Triton kernel.
    Fuses: ReLU + Flatten + Norm + Mul + Clamp + Div + Mul(weight)
    """
    # Get input shape: [B, C, H, W]
    B, C, H, W = in_1.shape
    spatial_size = H * W
    
    # Output shape after flatten: [B, C, H*W]
    output_shape = (B, C, spatial_size)
    
    # Allocate output
    output = torch.empty(output_shape, device=in_1.device, dtype=in_1.dtype)
    
    # Calculate grid
    # Each program handles one (batch, channel) pair
    num_programs = B * C
    BLOCK_SIZE = 1024
    
    # Round up spatial_size to nearest BLOCK_SIZE for kernel
    # But we handle masking inside the kernel
    
    # Launch kernel - use num_programs as grid
    fused_norm_kernel[(num_programs,)](
        in_1,           # input_ptr
        in_0,           # weight_ptr  
        output,         # output_ptr
        0.14433756729740643,  # norm_factor_ptr (passed as constant)
        B,              # batch_size
        C,              # channels
        spatial_size,   # spatial_size
        1e-05,          # EPS (for clamp)
        BLOCK_SIZE,     # BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return fused_norm_wrapper