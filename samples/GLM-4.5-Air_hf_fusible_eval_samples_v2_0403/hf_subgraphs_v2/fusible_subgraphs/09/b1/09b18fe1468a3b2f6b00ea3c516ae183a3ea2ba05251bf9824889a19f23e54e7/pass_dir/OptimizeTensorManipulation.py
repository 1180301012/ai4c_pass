import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, tmp_5):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_9 = tmp_5.unsqueeze(3)
    tmp_10 = tmp_8 - tmp_6
    return (tmp_10, tmp_9)

def replacement_args(in_4, in_0, tmp_5):
    return (in_4, in_0, tmp_5)

@triton.jit
def optimized_tensor_kernel(
    in_4_ptr, in_0_ptr, tmp_5_ptr, out_10_ptr, out_9_ptr,
    n_items_4096, n_dim_512, n_dim_32,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr, BLOCK_SIZE_Z: tl.constexpr
):
    # Program indices for 3D grid
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1) 
    pid_z = tl.program_id(2)
    
    # Create offsets for each dimension
    x_offsets = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_offsets = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    z_offsets = pid_z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
    
    # Create bounds checking masks
    x_mask = x_offsets < n_items_4096
    y_mask = y_offsets < n_dim_32
    z_mask = z_offsets < n_dim_512
    
    # Direct compute expanded tensor without explicit unsqueeze + expand
    # in_4: [1, 4096, 512] → when expanded → [1, 4096, 32, 512]
    # Directly access elements with broadcasting pattern
    expanded_4 = tl.load(in_4_ptr + (x_offsets[:, None] * 512 + z_offsets[None, :]),
                        mask=x_mask[:, None] & z_mask[None, :],
                        other=0.0)
    
    # in_0: [32, 512] → [1, 1, 32, 512] - view operation becomes direct access
    viewed_0 = tl.load(in_0_ptr + (y_offsets[:, None] * 512 + z_offsets[None, :]),
                       mask=y_mask[:, None] & z_mask[None, :],
                       other=0.0)
    
    # Compute subtraction: expanded_4 - viewed_0
    # Broadcasting: viewed_0 [32, 512] → [4096, 32, 512] 
    result_10 = expanded_4 - viewed_0[:, None, :]  # Broadcast along batch dimension
    
    # Compute tmp_9 expansion: tmp_5.unsqueeze(3)
    # tmp_5: [1, 4096, 32] → [1, 4096, 32, 1]
    result_9 = tl.load(tmp_5_ptr + (x_offsets[:, None] * 32 + y_offsets[None, :]),
                      mask=x_mask[:, None] & y_mask[None, :],
                      other=0.0)
    
    # Store results with appropriate offsets
    # Store tmp_10 result: [1, 4096, 32, 512]
    out_10_base = out_10_ptr + (x_offsets[:, None, None] * (n_items_4096 * n_dim_32 * n_dim_512) +
                                y_offsets[None, :, None] * (n_dim_32 * n_dim_512) +
                                z_offsets[None, None, :])
    tl.store(out_10_base, 
             result_10[:, None, :],
             mask=x_mask[:, None, None] & y_mask[None, :, None] & z_mask[None, None, :])
    
    # Store tmp_9 result: [1, 4096, 32, 1]
    out_9_base = out_9_ptr + (x_offsets[:, None, None] * (n_items_4096 * n_dim_32 * 1) +
                               y_offsets[None, :, None] * (n_dim_32 * 1) +
                               0)
    tl.store(out_9_base,
             result_9[:, None, :][:, :, :, None],  # Add singleton dimension at end
             mask=x_mask[:, None, None] & y_mask[None, :, None])

@torch.fx.wrap  
def optimized_tensor_manipulation(in_4, in_0, tmp_5):
    # Get tensor shapes
    in_4_shape = in_4.shape  # [1, 4096, 512]
    in_0_shape = in_0.shape  # [32, 512]
    tmp_5_shape = tmp_5.shape  # [1, 4096, 32]
    
    # Create output tensors
    tmp_6_shape = (1, 1, 32, 512)  # This is view of in_0
    tmp_10_shape = (1, 4096, 32, 512)  # Final subtraction result
    tmp_9_shape = (1, 4096, 32, 1)  # Softmax result with extra dimension
    
    # Reshape inputs to simplify kernel access patterns
    in_4_flat = in_4.reshape(4096, 512)  # Remove batch dimension for easier access
    in_0_flat = in_0.reshape(32, 512)    # Already correct shape for kernel
    tmp_5_flat = tmp_5.reshape(4096, 32) # Remove batch dimension
    
    # Create output tensors
    out_10 = torch.empty(tmp_10_shape, dtype=in_4.dtype, device=in_4.device)
    out_9 = torch.empty(tmp_9_shape, dtype=tmp_5.dtype, device=tmp_5.device)
    
    # Set block sizes for optimal GPU occupancy
    BLOCK_SIZE_X = 64   # Process 4096 items in chunks of 64
    BLOCK_SIZE_Y = 32   # Process 32 items in chunks of 32
    BLOCK_SIZE_Z = 128  # Process 512 items in chunks of 128
    
    # Calculate grid size
    grid_x = (4096 + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (32 + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_z = (512 + BLOCK_SIZE_Z - 1) // BLOCK_SIZE_Z
    
    # Launch 3D kernel
    optimized_tensor_kernel[(grid_x, grid_y, grid_z)](
        in_4_flat,
        in_0_flat,
        tmp_5_flat,
        out_10,
        out_9,
        4096, 512, 32,
        BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z
    )
    
    return (out_10, out_9)

def replacement_func():
    return optimized_tensor_manipulation