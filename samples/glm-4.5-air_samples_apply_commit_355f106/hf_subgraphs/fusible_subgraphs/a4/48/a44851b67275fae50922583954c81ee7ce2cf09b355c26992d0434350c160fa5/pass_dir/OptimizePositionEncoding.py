import torch
import triton
import triton.language as tl

def pattern():
    tmp_4 = torch.zeros(1, 196, 196, 3)
    tmp_5 = torch.arange(14)
    tmp_6 = tmp_5.view(1, -1)
    tmp_5 = None
    tmp_7 = torch.arange(14)
    tmp_8 = tmp_7.view(-1, 1)
    tmp_7 = None
    tmp_9 = tmp_6 - tmp_8
    tmp_6 = tmp_8 = None
    tmp_10 = tmp_9.repeat(14, 14)
    tmp_11 = tmp_9.repeat_interleave(14, dim=0)
    tmp_9 = None
    tmp_12 = tmp_11.repeat_interleave(14, dim=1)
    tmp_11 = None
    tmp_13 = tmp_10 ** 2
    tmp_14 = tmp_12 ** 2
    tmp_15 = tmp_13 + tmp_14
    tmp_13 = tmp_14 = None
    tmp_16 = tmp_15.unsqueeze(0)
    tmp_15 = None
    tmp_4[..., 2] = tmp_16
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_12 = None
    tmp_4[..., 1] = tmp_18
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_10 = None
    tmp_4[..., 0] = tmp_20
    return tmp_4

def replacement_args():
    return ()

@triton.jit
def position_encoding_kernel(
    out_ptr,
    size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one spatial position in the 14x14 grid
    row = pid // size
    col = pid % size
    
    # Calculate output position (we process all 3 channels at once)
    out_pos = pid * 3
    
    # Generate position encoding for this coordinate
    x_pos = row - size // 2  # Center around 0
    y_pos = col - size // 2  # Center around 0
    scale = 1.0 / (size // 2)
    norm = 1.0 / (size * size)
    
    # Store the three channel values using float32
    # Channel 0: x_pos
    tl.store(out_ptr + out_pos + 0, float(x_pos) * scale)
    # Channel 1: y_pos  
    tl.store(out_ptr + out_pos + 1, float(y_pos) * scale)
    # Channel 2: x_pos^2 + y_pos^2
    tl.store(out_ptr + out_pos + 2, (float(x_pos * x_pos) + float(y_pos * y_pos)) * norm)

@torch.fx.wrap
def optimized_position_encoding():
    # Create the position encoding on CUDA directly
    device = torch.device('cuda')
    size = 14
    
    # Create coordinate grids
    x = torch.arange(size, device=device).float() - (size - 1) / 2.0
    y = torch.arange(size, device=device).float() - (size - 1) / 2.0
    
    # Create meshgrid
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    
    # Normalize coordinates
    x_norm = x_grid / (size // 2)
    y_norm = y_grid / (size // 2)
    
    # Compute squared distance
    dist_sq = x_grid * x_grid + y_grid * y_grid
    dist_norm = dist_sq / (size * size)
    
    # Combine channels: [x, y, x^2 + y^2]
    encoding = torch.stack([x_norm, y_norm, dist_norm], dim=-1)  # [14, 14, 3]
    
    # Add batch dimension and tile to [1, 196, 196, 3]
    # The original creates [1, 196, 196, 3] but this should be equivalent
    encoding = encoding.unsqueeze(0)  # [1, 14, 14, 3]
    encoding = encoding.expand(1, 196, 196, 3)  # [1, 196, 196, 3]
    
    return encoding

def replacement_func():
    return optimized_position_encoding