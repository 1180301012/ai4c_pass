import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation pattern from model.py
    tmp_0 = torch.cat([in_1, in_0]);  in_1 = in_0 = None
    tmp_1 = torch.arange(32)
    tmp_2 = torch.arange(32)
    meshgrid = torch.meshgrid(tmp_1, tmp_2, indexing='ij');  tmp_1 = tmp_2 = None
    tmp_4 = meshgrid[0]
    tmp_5 = meshgrid[1];  meshgrid = None
    tmp_6 = torch.stack((tmp_4, tmp_5));  tmp_4 = tmp_5 = None
    tmp_7 = torch.flatten(tmp_6, 1);  tmp_6 = None
    tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
    tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))];  tmp_7 = None
    tmp_10 = tmp_8 - tmp_9;  tmp_8 = tmp_9 = None
    tmp_11 = tmp_10.permute(1, 2, 0);  tmp_10 = None
    tmp_12 = tmp_11.contiguous();  tmp_11 = None
    tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_13 += 31;  tmp_14 = tmp_13;  tmp_13 = None
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_14;  setitem = tmp_12;  tmp_14 = setitem = None
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += 31;  tmp_17 = tmp_16;  tmp_16 = None
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_17;  setitem_1 = tmp_12;  tmp_17 = setitem_1 = None
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= 63;  tmp_20 = tmp_19;  tmp_19 = None
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_20;  setitem_2 = tmp_12;  tmp_20 = setitem_2 = None
    return tmp_0, tmp_12

@triton.jit
def coordinate_generation_kernel(
    output_ptr,
    size,
    add_val1,
    add_val2,
    mul_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of the coordinate grid
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Generate relative coordinates
    coord_x = tl.arange(0, BLOCK_SIZE, dtype=tl.int32) + col
    coord_y = tl.arange(0, BLOCK_SIZE, dtype=tl.int32) + row
    
    # Create meshgrid of coordinates
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            rel_x = coord_x[j] - coord_x[0]
            rel_y = coord_y[i] - coord_y[0]
            
            # Apply transformations
            trans_x = rel_x + add_val1
            trans_x = trans_x * mul_val
            trans_y = rel_y + add_val2
            
            # Store results
            offset = ((i + row * BLOCK_SIZE) * size + (j + col * BLOCK_SIZE)) * 2
            tl.store(output_ptr + offset, trans_x)
            tl.store(output_ptr + offset + 1, trans_y)

@triton.jit
def optimized_coordinate_kernel(
    output_ptr,
    size,
    add_val1,
    add_val2,
    mul_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized coordinate generation with better memory access patterns
    block_id = tl.program_id(0)
    total_elements = size * size
    elements_per_block = BLOCK_SIZE * 2  # Each element has x,y coordinates
    
    start_idx = block_id * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, total_elements)
    
    for idx in range(start_idx, end_idx):
        row = idx // size
        col = idx % size
        
        # Generate coordinates
        coord_y = row
        coord_x = col
        
        # Apply transformations
        trans_x = coord_x + add_val1
        trans_x = trans_x * mul_val
        trans_y = coord_y + add_val2
        
        # Store results
        offset = idx * 2
        tl.store(output_ptr + offset, trans_x)
        tl.store(output_ptr + offset + 1, trans_y)

@torch.fx.wrap
def optimized_coordinate_generation(in_0, in_1, size, add_val1, add_val2, mul_val):
    # Concatenate inputs using allowed APIs only
    input_shape_1 = in_1.shape
    input_shape_0 = in_0.shape
    total_rows = input_shape_0[0] + input_shape_1[0]
    concat_shape = (total_rows, input_shape_0[1])
    tmp_0 = torch.empty(concat_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Copy in_1 data
    tmp_0[:input_shape_1[0]] = in_1
    # Copy in_0 data  
    tmp_0[input_shape_1[0]:] = in_0
    
    # Create coordinate tensor
    coord_shape = (size * size, 2)
    output = torch.empty(coord_shape, dtype=torch.int64, device=in_0.device)
    
    # Launch optimized kernel
    BLOCK_SIZE = 1024
    total_elements = size * size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_coordinate_kernel[(num_programs,)](
        output_ptr=output,
        size=size,
        add_val1=add_val1,
        add_val2=add_val2,
        mul_val=mul_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_0, output

def replacement_args(in_0, in_1):
    size = 32  # Default for this pattern, can be parameterized
    add_val1 = 31
    add_val2 = 31  
    mul_val = 63
    return in_0, in_1, size, add_val1, add_val2, mul_val

def replacement_func():
    return optimized_coordinate_generation