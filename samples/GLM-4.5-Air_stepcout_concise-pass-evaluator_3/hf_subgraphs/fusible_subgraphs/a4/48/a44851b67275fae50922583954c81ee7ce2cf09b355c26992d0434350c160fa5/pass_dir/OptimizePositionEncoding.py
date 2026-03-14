import torch
import triton
import triton.language as tl

# Pattern matching function for position encoding generation
def pattern():
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
    tmp_4 = torch.zeros(1, 196, 196, 3)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_16
    tmp_17 = tmp_4
    tmp_16 = tmp_17 = None
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_12 = None
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_18
    tmp_19 = tmp_4
    tmp_18 = tmp_19 = None
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_10 = None
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_20
    tmp_21 = tmp_4
    tmp_20 = tmp_21 = None
    return tmp_4

# Argument extraction function (no inputs needed)
def replacement_args():
    return ()

# Triton kernel for optimized position encoding generation
@triton.jit
def position_encoding_kernel(
    out_ptr,
    grid_h: tl.constexpr,
    grid_w: tl.constexpr,
    base_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= grid_h * grid_w:
        return
    
    # Calculate grid coordinates
    y = pid // grid_w
    x = pid % grid_w
    
    # Calculate base coordinates (0 to 13)
    base_y = y // base_size
    base_x = x // base_size
    
    # Compute coordinate differences
    rel_y = base_y - base_x
    
    # Generate three channels:
    # Channel 0: base_y coordinates (broadcasted)
    # Channel 1: base_x coordinates (broadcasted) 
    # Channel 2: squared euclidean distance
    # Convert to float32 for storage
    channel0 = tl.cast(base_y, tl.float32)
    channel1 = tl.cast(base_x, tl.float32)
    channel2 = tl.cast(rel_y * rel_y, tl.float32)
    
    # Calculate output position  
    pos = y * grid_w + x
    
    # Store all three channels
    tl.store(out_ptr + pos * 3 + 0, channel0)
    tl.store(out_ptr + pos * 3 + 1, channel1)
    tl.store(out_ptr + pos * 3 + 2, channel2)

@torch.fx.wrap
def optimized_position_encoding():
    # grid_size = 196  # 14x14 repeated
    base_size = 14   # Original arange size
    channels = 3
    
    # grid_size = 196  # 14x14 repeated
    out = torch.empty((1, 196, 196, channels), dtype=torch.float32, device='cuda:0')
    
    BLOCK_SIZE = 1024
    num_programs = (196 * 196 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    position_encoding_kernel[(num_programs,)](
        out,
        grid_h=196,
        grid_w=196,
        base_size=base_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_position_encoding