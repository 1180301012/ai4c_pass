import torch
import triton
import triton.language as tl

def pattern(tmp_5):
    """
    Pattern: View -> Transpose -> Contiguous -> View (Channel Shuffle)
    Match for: tmp_5 -> view -> transpose -> contiguous -> view
    """
    tmp_7 = tmp_5.view(256, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(256, 40, 64, 48)
    return tmp_10

def replacement_args(tmp_5):
    return (tmp_5,)

@triton.jit
def channel_shuffle_kernel(
    input_ptr, output_ptr,
    B: tl.constexpr, G: tl.constexpr, C_per_G: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized channel shuffle kernel
    Input/Output shape: [B, G*C_per_G, H, W]
    
    The operation transforms: view(B,G,C_per_G,H,W) -> transpose(1,2) -> view(B,G*C_per_G,H,W)
    This means: input[b,g*C_per_G+c,h,w] -> output[b,c*G+g,h,w]
    """
    pid = tl.program_id(0)
    
    # Total number of elements
    total_elements = B * G * C_per_G * H * W
    
    # Calculate offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decode output indices: [b, c_out, h, w]
    b_out = offsets // (G * C_per_G * H * W)
    remainder = offsets % (G * C_per_G * H * W)
    c_out = remainder // (H * W)
    remainder = remainder % (H * W)
    h_out = remainder // W
    w_out = remainder % W
    
    # Map output channel to input channel
    # After transpose(1,2): (B, G, C_per_G, H, W) -> (B, C_per_G, G, H, W)
    # So output[b, c_per_g * G + g, h, w] comes from input[b, g * C_per_G + c_per_g, h, w]
    c_per_g_idx = c_out // G
    g_idx = c_out % G
    
    # Input channel index
    c_in = g_idx * C_per_G + c_per_g_idx
    
    # Input linear index
    input_idx = b_out * (G * C_per_G * H * W) + c_in * (H * W) + h_out * W + w_out
    
    # Load from input
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Store to output
    tl.store(output_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_channel_shuffle_256_2_20_64_48(input_tensor):
    """
    Fused channel shuffle for shape [256, 40, 64, 48] with G=2, C_per_G=20
    """
    B, C, H, W = input_tensor.shape
    G = 2
    C_per_G = C // G  # Should be 20
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    total_elements = B * C * H * W
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    channel_shuffle_kernel[grid](
        input_tensor, output,
        B, G, C_per_G, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_channel_shuffle_256_2_20_64_48