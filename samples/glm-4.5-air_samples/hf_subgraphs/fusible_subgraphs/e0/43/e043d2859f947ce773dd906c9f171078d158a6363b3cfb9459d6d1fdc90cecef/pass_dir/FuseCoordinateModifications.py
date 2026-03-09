import torch
import triton
import triton.language as tl

def pattern(tmp_12):
    # Extract coordinate dimensions and apply transformations
    # tmp_12 has shape (24, 24, 2) or (32, 32, 2)
    
    # Extract and modify first coordinate channel
    tmp_13 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_14 = tmp_13 + 23
    
    # Extract and modify second coordinate channel  
    tmp_16 = tmp_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_17 = tmp_16 + 23
    
    # Combine results and apply multiplication to first channel
    tmp_18 = torch.stack([tmp_14, tmp_17], dim=2)
    tmp_19 = tmp_18[slice(None, None, None), slice(None, None, None), 0]
    tmp_20 = tmp_19 * 47
    
    # Final tensor with modified coordinates
    result = torch.stack([tmp_20, tmp_18[slice(None, None, None), slice(None, None, None), 1]], dim=2)
    
    return result

@triton.jit
def coordinate_mod_kernel(
    coord_ptr,
    size,
    add_value_1: tl.constexpr,
    add_value_2: tl.constexpr,
    mul_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get global index
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < (size * size * 2)  # 2 channels
    
    # Load all coordinate data at once
    coords = tl.load(coord_ptr + pid, mask=mask, other=0)
    
    # Apply coordinate transformations based on channel
    # Channel 0: add_value_1 then multiply, Channel 1: just add_value_2
    channel_mask = (pid % (size * size)) < (size * size)  # Within current channel block
    
    # Extract channel info
    total_coords = size * size * 2
    half_coords = total_coords // 2
    
    # Channel 0 operations (add + multiply)
    chan0_mask = (pid < half_coords)
    coords_ch0 = tl.where(chan0_mask, 
                          coords + add_value_1,
                          coords)
    coords_ch0 = tl.where(chan0_mask, 
                          coords_ch0 * mul_value,
                          coords_ch0)
    
    # Channel 1 operations (just add)
    chan1_mask = ((pid >= half_coords) & (pid < total_coords))
    coords_ch1 = tl.where(chan1_mask,
                          coords + add_value_2,
                          coords)
    
    # Combine results
    final_coords = tl.where(chan0_mask, coords_ch0, 
                           tl.where(chan1_mask, coords_ch1, coords))
    
    # Store results
    tl.store(coord_ptr + pid, final_coords, mask=mask)

@triton.jit
def coordinate_mod_split_kernel(
    coord0_ptr,
    coord1_ptr,
    size,
    add_val_0,
    add_val_1,
    mul_val_0,
    BLOCK_SIZE: tl.constexpr,
):
    # Get global index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (size * size)
    
    # Load channel 0 and channel 1 data
    chan0 = tl.load(coord0_ptr + idx, mask=mask, other=0)
    chan1 = tl.load(coord1_ptr + idx, mask=mask, other=0)
    
    # Apply transformations
    chan0_updated = (chan0 + add_val_1) * mul_val_0
    chan1_updated = chan1 + add_val_1
    
    # Store results
    tl.store(coord0_ptr + idx, chan0_updated, mask=mask)
    tl.store(coord1_ptr + idx, chan1_updated, mask=mask)

@torch.fx.wrap
def modify_coordinates_optimized(tmp_12, size, add_val, mul_val):
    # Get tensor properties
    coords_ptr = tmp_12.data_ptr()
    device = tmp_12.device
    
    # Optimized kernel that processes both channels simultaneously
    BLOCK_SIZE = 1024
    total_elements = size * size * 2
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    coordinate_mod_kernel[(num_programs,)](
        coords_ptr,
        size,
        add_val,  # add_value_1 
        add_val,  # add_value_2
        mul_val,  # mul_value
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return tmp_12

# Wrapper with specific values for this graph
@torch.fx.wrap  
def coordinate_mod_ops_base(tmp_12):
    return modify_coordinates_optimized(tmp_12, size=24, add_val=23, mul_val=47)

@torch.fx.wrap
def coordinate_mod_ops_large(tmp_12):
    return modify_coordinates_optimized(tmp_12, size=32, add_val=31, mul_val=63)

def replacement_args(tmp_12):
    return (tmp_12,)

def replacement_func():
    # Return the function for the base graph (size=24)
    return coordinate_mod_ops_base