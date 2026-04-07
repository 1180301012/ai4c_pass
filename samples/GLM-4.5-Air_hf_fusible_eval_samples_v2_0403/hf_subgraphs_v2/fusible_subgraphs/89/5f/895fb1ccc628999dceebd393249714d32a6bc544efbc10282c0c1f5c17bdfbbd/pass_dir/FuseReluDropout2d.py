import torch
import triton
import triton.language as tl

# Pattern matching function - matches ReLU + DropOut2d sequence
def pattern(tmp_3):
    # This matches: relu_ -> dropout2d sequence
    tmp_4 = torch.relu_(tmp_3)
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return tmp_5

# Argument extraction function  
def replacement_args(tmp_3):
    return (tmp_3,)

@triton.jit
def optimized_relu_dropout2d_kernel(
    input_ptr,      # [1, 512, 64, 64] - input to ReLU
    output_ptr,     # [1, 512, 64, 64] - output after ReLU + dropout
    n_channels,     # 512
    height,         # 64
    width,          # 64
    dropout_prob,   # 0.1
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a spatial tile
    program_id = tl.program_id(0)
    spatial_elems = height * width
    
    # Compute grid dimensions for spatial tiling
    tiles_per_dim = triton.cdiv(spatial_elems, BLOCK_SIZE)
    tile_id = program_id
    
    if tile_id >= tiles_per_dim:
        return
    
    # Compute tile boundaries
    tile_start = tile_id * BLOCK_SIZE
    tile_end = min(tile_start + BLOCK_SIZE, spatial_elems)
    
    # Process the entire spatial tile for all channels
    for spatial_offset in range(tile_start, tile_end):
        # Compute 2D coordinates
        h_offset = spatial_offset // width
        w_offset = spatial_offset % width
        
        # Process all channels for this spatial location
        for c in range(n_channels):
            # Calculate memory offset
            input_offset = (0 * n_channels * height * width) + (c * height * width) + (h_offset * width) + w_offset
            output_offset = input_offset  # Same shape
            
            # Load input value
            inp_val = tl.load(input_ptr + input_offset)
            
            # Apply ReLU
            relu_val = tl.maximum(inp_val, 0.0)
            
            # Apply dropout (only during training simulation)
            # For this optimization, we keep the dropout logic as it would be
            # in a real training scenario
            dropout_scale = 1.0 / (1.0 - dropout_prob)  # scaling for training
            dropout_mask = tl.random.uniform() > dropout_prob
            out_val = relu_val * dropout_mask * dropout_scale
            
            # Store result
            tl.store(output_ptr + output_offset, out_val)

@torch.fx.wrap  
def optimized_relu_dropout2d(input_tensor):
    """Optimized fusion of relu_ -> dropout2d operations"""
    # Input shapes
    n_channels = input_tensor.shape[1]  # 512
    height = input_tensor.shape[2]       # 64
    width = input_tensor.shape[3]        # 64
    spatial_elems = height * width
    
    # Output tensor
    out = torch.empty_like(input_tensor)
    
    # Block size for spatial processing
    BLOCK_SIZE = 1024  # Large block for spatial processing
    
    # Grid configuration: one thread per spatial tile
    grid_size = triton.cdiv(spatial_elems, BLOCK_SIZE)
    
    # Launch kernel
    optimized_relu_dropout2d_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=out,
        n_channels=n_channels,
        height=height,
        width=width,
        dropout_prob=0.1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - returns the optimized kernel implementation
def replacement_func():
    return optimized_relu_dropout2d