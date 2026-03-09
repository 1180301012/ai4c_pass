import torch
import triton
import triton.language as tl

# Pattern matching function - matches attention mask preprocessing
def pattern(x, y):
    # Very simple pattern to start with
    tmp_1 = y.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    return tmp_3, x


# Argument extraction function
def replacement_args(x, y):
    return (x, y)


# Optimized kernel for fused attention mask preprocessing (seq_len=1024)
@triton.jit
def attention_mask_kernel(
    attention_mask_ptr,
    scaled_mask_ptr,
    position_ids_ptr,
    sliced_position_ids_ptr,
    mask_shape_0: tl.constexpr,
    mask_shape_1: tl.constexpr,
    mask_shape_2: tl.constexpr,
    mask_shape_3: tl.constexpr,
    position_ids_shape_0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread IDs
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process mask data
    mask_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_mask = mask_offsets < (mask_shape_0 * mask_shape_1 * mask_shape_2 * mask_shape_3)
    mask_flat_idx = mask_offsets
    
    # Convert mask indexing from flattened to multi-dimensional
    mask_i = mask_flat_idx // (mask_shape_1 * mask_shape_2 * mask_shape_3)
    mask_j = (mask_flat_idx % (mask_shape_1 * mask_shape_2 * mask_shape_3)) // (mask_shape_2 * mask_shape_3)
    mask_k = (mask_flat_idx % (mask_shape_2 * mask_shape_3)) // mask_shape_3
    mask_l = mask_flat_idx % mask_shape_3
    
    # Load attention mask (int64)
    mask_val = tl.load(attention_mask_ptr + mask_offsets, mask=mask_mask, other=0)
    
    # Convert to float32, invert, and scale in one operation
    # mask is typically 1, so 1.0 - 1.0 = 0.0, then * -FLT_MAX = -inf
    scaled_val = (1.0 - tl.cast(mask_val, tl.float32)) * -3.4028234663852886e+38
    
    # Store scaled mask
    tl.store(scaled_mask_ptr + mask_offsets, scaled_val, mask=mask_mask)
    
    # Process position IDs slicing - only slice along the last dimension
    if pid == 0:  # Only first block handles position IDs slicing
        # Load original position IDs  
        original_pos_ids = tl.load(position_ids_ptr)
        
        # Create sliced position IDs - only take first 1024 elements along last dimension
        for i in range(position_ids_shape_0):
            for j in range(1024):
                if j < position_ids_shape[1]:  # Check bounds
                    src_idx = i * position_ids_shape[1] + j
                    tl.store(sliced_position_ids_ptr + i * 1024 + j, original_pos_ids[src_idx])


@torch.fx.wrap
def fused_attention_mask_processing(x, y):
    # Get tensor shapes
    mask_shape = y.shape
    position_ids_shape = x.shape
    
    # Create output tensors
    scaled_mask = torch.empty(mask_shape, dtype=torch.float32, device=y.device)
    sliced_position_ids = torch.empty([position_ids_shape[0], 1024], dtype=x.dtype, device=x.device)
    
    # Calculate grid size
    mask_elements = mask_shape[0] * mask_shape[1] * mask_shape[2] * mask_shape[3]
    BLOCK_SIZE = 1024
    grid_size = (mask_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    attention_mask_kernel[grid_size](
        attention_mask_ptr=y,
        scaled_mask_ptr=scaled_mask,
        position_ids_ptr=x,
        sliced_position_ids_ptr=sliced_position_ids,
        mask_shape_0=mask_shape[0],
        mask_shape_1=mask_shape[1],
        mask_shape_2=mask_shape[2],
        mask_shape_3=mask_shape[3],
        position_ids_shape_0=position_ids_shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return scaled_mask, sliced_position_ids


# Replacement function
def replacement_func():
    return fused_attention_mask_processing