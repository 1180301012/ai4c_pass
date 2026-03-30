import torch
import triton
import triton.language as tl

# Pattern matching function - optimize just the multiply-add part
def pattern(tmp_1, in_1):
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    return tmp_3  # Return observable tensor - tmp_3 is used for ReLU

# Argument extraction function
def replacement_args(in_0, in_1):
    # The original computation creates tmp_1 first, then calls the pattern
    tmp_1 = torch.sigmoid(in_0).view(1, 512, 1, 1)
    # Flatten tmp_1 to make it easier to access in kernel - this preserves the broadcastable values
    tmp_1_flat = tmp_1.reshape(512)  # [512] 
    return (tmp_1_flat, in_1)

# Simple Triton kernel for multiply-add fusion: tmp_3 = in_1 + (in_1 * tmp_1)
@triton.jit
def multiply_add_kernel(
    tmp1_ptr,  # [512] - flattened gating factor values
    in1_ptr,   # [1, 512, 64, 64] - main feature map
    out3_ptr,  # [1, 512, 64, 64] - output = in_1 + (in_1 * tmp_1)
    N,  # Total number of elements
    C: tl.constexpr,  # Channels = 512
    H: tl.constexpr,  # Height = 64
    W: tl.constexpr,  # Width = 64
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load main feature map
    in1_val = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate which channel each offset corresponds to
    spatial_channel_indices = offsets % C
    
    # Load gating factors for the channels we need from the flattened array
    channel_offsets = spatial_channel_indices
    channel_mask = channel_offsets < C
    tmp1_vals = tl.load(tmp1_ptr + channel_offsets, mask=channel_mask, other=0.0)
    
    # Compute: in_1 + (in_1 * tmp_1) = in_1 * (1 + tmp_1)
    result = in1_val * (1.0 + tmp1_vals)
    
    # Store result
    tl.store(out3_ptr + offsets, result, mask=mask)

# Kernel wrapper that handles the full computation
@torch.fx.wrap
def multiply_add_forward(tmp_1, in_1):
    # Calculate shapes
    _, C_in, H, W = in_1.shape  # [1, 512, 64, 64]
    
    N = C_in * H * W  # Total elements: 512 * 64 * 64 = 2097152
    
    # Convert to float32 for numerical stability
    tmp_1_fp32 = tmp_1.to(torch.float32)
    in_1_fp32 = in_1.to(torch.float32)
    
    # Allocate output tensor in float32
    tmp_3_out_fp32 = torch.empty_like(in_1_fp32)
    
    # Optimized grid/block configuration
    BLOCK_SIZE = 1024  # Optimal for 64x64 spatial dimensions
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the multiply-add kernel
    multiply_add_kernel[(num_programs,)](
        tmp1_ptr=tmp_1_fp32,
        in1_ptr=in_1_fp32,
        out3_ptr=tmp_3_out_fp32,
        N=N,
        C=C_in,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert back to original dtype
    tmp_3_out = tmp_3_out_fp32.to(in_1.dtype)
    
    return tmp_3_out  # Return tensor to match the expected output format

# Replacement function (returns function reference, not a call)
def replacement_func():
    return multiply_add_forward