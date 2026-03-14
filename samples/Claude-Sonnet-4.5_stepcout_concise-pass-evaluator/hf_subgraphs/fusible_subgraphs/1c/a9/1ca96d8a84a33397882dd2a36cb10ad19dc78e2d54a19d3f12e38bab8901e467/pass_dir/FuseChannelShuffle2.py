import torch
import triton
import triton.language as tl

def pattern(tmp_6):
    """
    Pattern: View -> Transpose -> Contiguous -> View (Channel Shuffle) for second branch
    """
    tmp_11 = tmp_6.view(256, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(256, 80, 32, 24)
    return tmp_14

def replacement_args(tmp_6):
    return (tmp_6,)

@triton.jit
def channel_shuffle_kernel_2(
    input_ptr, output_ptr,
    B: tl.constexpr, G: tl.constexpr, C_per_G: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = B * G * C_per_G * H * W
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    b_out = offsets // (G * C_per_G * H * W)
    remainder = offsets % (G * C_per_G * H * W)
    c_out = remainder // (H * W)
    remainder = remainder % (H * W)
    h_out = remainder // W
    w_out = remainder % W
    
    c_per_g_idx = c_out // G
    g_idx = c_out % G
    c_in = g_idx * C_per_G + c_per_g_idx
    input_idx = b_out * (G * C_per_G * H * W) + c_in * (H * W) + h_out * W + w_out
    
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_channel_shuffle_256_2_40_32_24(input_tensor):
    B, C, H, W = input_tensor.shape
    G = 2
    C_per_G = C // G
    output = torch.empty_like(input_tensor)
    total_elements = B * C * H * W
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    channel_shuffle_kernel_2[grid](input_tensor, output, B, G, C_per_G, H, W, BLOCK_SIZE=BLOCK_SIZE)
    return output

def replacement_func():
    return fused_channel_shuffle_256_2_40_32_24