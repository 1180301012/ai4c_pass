import torch
import triton
import triton.language as tl

def pattern(table, indices):
    """Pattern: gather -> view -> permute -> contiguous -> unsqueeze fusion"""
    tmp_2 = table[indices]                    # Gather operation
    tmp_3 = tmp_2.view(197, 197, -1)          # Reshape to 3D
    tmp_4 = tmp_3.permute(2, 0, 1)            # Permute dimensions to [C, H, W]
    tmp_5 = tmp_4.contiguous()                # Ensure memory contiguity
    tmp_6 = tmp_5.unsqueeze(0)                # Add batch dimension
    return tmp_6

def replacement_args(table, indices):
    return (table, indices)

@triton.jit
def fused_gather_reshape_kernel(
    table_ptr,
    indices_ptr,
    output_ptr,
    table_cols,
    table_rows,  # Add table_rows parameter
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: gather indices, reshape, permute, and add batch dimension in one pass"""
    # Each program handles one element in the final [C, H, W] layout (without batch dimension)
    program_id = tl.program_id(0)
    
    # Total elements in final output: C * H * W (without batch dimension)
    if program_id >= num_elements:
        return
    
    # For safety: validate input shapes
    if table_cols <= 0:
        return
    if table_rows <= 0:
        return
    if num_elements <= 0:
        return
    
    # Calculate coordinates in final [C, H, W] layout
    # w, h, c represent coordinates in [channels, height, width] space
    w_coord = program_id % 197                      # Width coordinate (0-196)
    h_coord = (program_id // 197) % 197            # Height coordinate (0-196)  
    c_coord = program_id // (197 * 197)            # Channel coordinate
    
    # The original transformation sequence:
    # 1. Gather from table: table[indices] -> flat array of 38809 elements
    # 2. view(197, 197, -1) -> [197, 197, table_cols]
    # 3. permute(2, 0, 1) -> rearranges dimensions to [table_cols, 197, 197]
    # 4. unsqueeze(0) -> [1, table_cols, 197, 197]
    
    # Reverse mapping from final [C, H, W] back to original [H, W, C]:
    # Original transform: [H, W, C] -> permute(2, 0, 1) -> [C, H, W]
    # permute(2, 0, 1) means: 
    #   new_axis0 = original_axis2 (C)
    #   new_axis1 = original_axis0 (H) 
    #   new_axis2 = original_axis1 (W)
    # So to reverse: [C, H, W] -> [H, W, C] requires:
    #   original_H = new_H (which is h_coord)
    #   original_W = new_W (which is w_coord) 
    #   original_C = new_C (which is c_coord)
    
    original_h = h_coord
    original_w = w_coord
    original_c = c_coord
    
    # Calculate linear index in the flat gathered array [197 * 197 * table_cols]
    flat_index = original_h * 197 * table_cols + original_w * table_cols + original_c
    
    # Ensure flat_index is within bounds
    if flat_index >= num_elements:
        # For debugging: print safe value instead of crashing
        tl.store(output_ptr + program_id, 0.0)
        return
    
    # The indices array contains values that are into [table_rows, table_cols]
    # Load the index value
    index = tl.load(indices_ptr + flat_index)
    
    # Calculate position in the original table
    table_row = index // table_cols
    table_col = index % table_cols
    
    # Ensure indices are valid before accessing table (fix chained boolean operators)
    if table_row >= table_rows:
        # Safe fallback: store zero
        tl.store(output_ptr + program_id, 0.0)
        return
    if table_col >= table_cols:
        # Safe fallback: store zero
        tl.store(output_ptr + program_id, 0.0)
        return
    
    # Calculate offset in the table
    table_offset = table_row * table_cols + table_col
    
    # Load value from table (ensure bounds)
    result = tl.load(table_ptr + table_offset)
    
    # Store result to output at program_id position
    tl.store(output_ptr + program_id, result)

@torch.fx.wrap
def fused_gather_reshape_unsqueeze(table, indices):
    """Optimized fused kernel for gather + reshape + permute + unsqueeze"""
    # Get dimensions
    table_cols = table.shape[1]
    table_rows = table.shape[0]
    
    # Final shape: [1, table_cols, 197, 197]
    batch_size = 1
    channels = table_cols  
    height = 197
    width = 197
    total_elements = channels * height * width
    
    # Create output tensor with correct shape
    output = torch.empty([batch_size, channels, height, width], 
                        dtype=table.dtype, device=table.device)
    
    # Flatten output for easier kernel access (temporarily)
    output_flat = output.reshape(-1)  # [channels * height * width]
    
    # Launch optimized kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    fused_gather_reshape_kernel[grid](
        table_ptr=table,
        indices_ptr=indices,
        output_ptr=output_flat,
        table_cols=table_cols,
        table_rows=table_rows,
        num_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_gather_reshape_unsqueeze