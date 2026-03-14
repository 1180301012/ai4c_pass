import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Matches the softmax -> multiply -> reduce_sum pattern
    """
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for fused softmax, multiply, and reduce_sum
@triton.jit
def fused_softmax_multiply_reduce_sum_kernel(
    in_0_ptr,  # Input feature map [B, C, D, H, W]
    in_1_ptr,  # Input attention weights [B, C, D, 1, 1]
    out_ptr,   # Output [B, D, H, W]
    B, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Program identifiers - one program per (d, h, w) position
    pid = tl.program_id(0)
    
    # Extract depth and spatial position from program ID
    total_spatial = H * W
    d = pid // total_spatial  # depth position  
    hw_id = pid % total_spatial  # spatial position
    h = hw_id // W             # height position
    w = hw_id % W              # width position
    
    # Each program handles one block of channels
    c_offset = tl.arange(0, BLOCK_SIZE)
    channel_mask = c_offset < C
    
    # Calculate memory offsets for batch=0 (since batch size is 1)
    # For both tensors, the batch dimension has size 1
    
    # For in_0: [1, C, D, H, W] -> offsets for each channel at current (d, h, w)
    in_0_base = c_offset * (D * H * W) + d * (H * W) + h * W + w
    
    # For in_1: [1, C, D, 1, 1] -> offsets for each channel at current (d, 0, 0) 
    # since in_1 only has [1, 1] spatial dimensions
    in_1_base = c_offset * (D * 1 * 1) + d * (1 * 1) + 0 * 1 + 0
    
    # Load feature map values and attention weights for current channels and positions
    features = tl.load(in_0_ptr + in_0_base, mask=channel_mask, other=0.0)
    attn_weights = tl.load(in_1_ptr + in_1_base, mask=channel_mask, other=-float('inf'))
    
    # Apply softmax to attention weights across channels (axis=0)
    max_val = tl.max(attn_weights, axis=0)
    shifted = attn_weights - max_val
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    softmax_weights = exp_shifted / sum_exp
    
    # Multiply features with attention weights and sum across channels
    weighted_features = features * softmax_weights
    summed = tl.sum(weighted_features, axis=0)
    
    # Store result in output tensor at [B, D, H, W] position
    out_base = d * (H * W) + h * W + w  # Skip channel dimension for output
    tl.store(out_ptr + out_base, summed)

# Kernel wrapper
@torch.fx.wrap
def fused_softmax_multiply_reduce_sum(in_0, in_1):
    B, C, D, H, W = in_0.shape
    B_in_1, C_in_1, D_in_1, H_in_1, W_in_1 = in_1.shape
    
    # Create broadcasted version of in_1 to match in_0 shape for kernel
    if B_in_1 == 1 and C_in_1 == 1 and D_in_1 == D and H_in_1 == H and W_in_1 == W:
        # If shapes are compatible, PyTorch will handle broadcasting automatically
        pass  # We'll let kernel handle broadcasting through memory layout
    
    # Output shape after sum reduction on dimension 1: [B, D, H, W]
    out = torch.empty((B, D, H, W), dtype=in_0.dtype, device=in_0.device)
    
    # Adjust BLOCK_SIZE for better GPU occupancy
    if C <= 256:
        BLOCK_SIZE = min(C, 128)
    else:
        BLOCK_SIZE = 256
    
    # Calculate grid dimensions: one program per (d, h, w) position
    total_positions = D * H * W
    grid_size = (total_positions + 1023) // 1024
    
    # Launch kernel (1D grid since pid handles all positions)
    fused_softmax_multiply_reduce_sum_kernel[(grid_size), 1, 1](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B, C=C, D=D, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_softmax_multiply_reduce_sum