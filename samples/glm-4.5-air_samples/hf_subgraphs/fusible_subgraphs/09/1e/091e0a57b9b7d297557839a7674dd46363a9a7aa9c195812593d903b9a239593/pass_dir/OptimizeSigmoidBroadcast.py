import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    tmp_0 = c.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(b)
    tmp_3 = b * tmp_2
    tmp_3 += a
    return tmp_3

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def sigmoid_broadcast_kernel(
    in_2_ptr,
    out_ptr,
    num_channels,
    height,
    width,
    BLOCK_SIZE_channels: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    # Calculate grid positions
    c = tl.program_id(0) * BLOCK_SIZE_channels + tl.arange(0, BLOCK_SIZE_channels)
    h = tl.program_id(1) * BLOCK_HEIGHT + tl.arange(0, BLOCK_HEIGHT)
    w = tl.program_id(2) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    
    # Create masks for bounds checking
    c_mask = c < num_channels
    h_mask = h < height
    w_mask = w < width
    
    # Combine masks
    mask = c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    
    # Load sigmoid input (all spatial positions use the same value from in_2)
    # in_2 has shape [1, 1, num_channels]
    sigmoid_input = tl.load(in_2_ptr + c[None, None, :], mask=c_mask[None, None, :], other=0.0)
    
    # Broadcast sigmoid across spatial dimensions
    sigmoid_result = tl.sigmoid(sigmoid_input)
    
    # Store result - broadcast to all spatial positions
    ptr_offsets = c[None, None, :] * (height * width) + h[None, :, None] * width + w[None, None, :]
    tl.store(out_ptr + ptr_offsets, sigmoid_result, mask=mask)

@torch.fx.wrap
def optimized_sigmoid_broadcast(in_2, target_shape):
    num_channels, height, width = target_shape[1], target_shape[2], target_shape[3]
    
    # Determine optimal block sizes based on tensor sizes
    BLOCK_SIZE_channels = min(2048, num_channels)
    BLOCK_HEIGHT = 16
    BLOCK_WIDTH = 16
    
    # Calculate grid dimensions
    num_channel_programs = (num_channels + BLOCK_SIZE_channels - 1) // BLOCK_SIZE_CHANNELS
    num_height_programs = (height + BLOCK_HEIGHT - 1) // BLOCK_HEIGHT
    num_width_programs = (width + BLOCK_WIDTH - 1) // BLOCK_WIDTH
    
    # Create output tensor
    out = torch.empty(target_shape, dtype=torch.float32, device=in_2.device)
    
    # Launch kernel
    sigmoid_broadcast_kernel[(num_channel_programs, num_height_programs, num_width_programs)](
        in_2_ptr=in_2,
        out_ptr=out,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE_channels=BLOCK_SIZE_channels,
        BLOCK_HEIGHT=BLOCK_HEIGHT,
        BLOCK_WIDTH=BLOCK_WIDTH,
    )
    
    return out

def replacement_func():
    def optimized_fused_operation(in_0, in_1, in_2):
        # Compute sigmoid broadcast using our optimized kernel
        sigmoid_broadcasted = optimized_sigmoid_broadcast(in_2, in_1.shape)
        # Fused element-wise operations: in_1 * sigmoid + in_0
        return in_1 * sigmoid_broadcasted + in_0
    return optimized_fused_operation