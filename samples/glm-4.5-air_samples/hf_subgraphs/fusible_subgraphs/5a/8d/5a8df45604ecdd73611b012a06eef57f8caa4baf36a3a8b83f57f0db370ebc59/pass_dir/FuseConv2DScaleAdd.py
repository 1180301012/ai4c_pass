import torch
import triton
import triton.language as tl

# Pattern matching function that matches the conv2d + scaling + add sequence
def pattern(x, weight, bias, scale, residual):
    # Conv2D operation with 1x1 kernel
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Scaling (broadcasting scale factor to match conv output)
    # For 1x1 conv, scale should have shape [C_out] and broadcast to spatial dimensions
    scaled_out = conv_out * scale.unsqueeze(-1).unsqueeze(-1)
    
    # Addition (residual connection)
    result = residual + scaled_out
    return result

# Argument extraction function
def replacement_args(x, weight, bias, scale, residual):
    return (x, weight, bias, scale, residual)

# Optimized kernel for fused conv2d + scaling + add
@triton.jit
def fused_conv_scale_add_kernel(
    x_ptr,                 # Input tensor [B, C_in, H, W]
    weight_ptr,            # Conv weights [C_out, C_in, 1, 1]  
    bias_ptr,              # Conv bias [C_out]
    scale_ptr,             # Scaling factor [C_out]
    residual_ptr,          # Residual tensor [B, C_out, H, W]
    out_ptr,               # Output tensor [B, C_out, H, W]
    B, C_in, C_out, H, W,  # Tensor dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program IDs - 2D grid: (batch, output_channel)
    batch_id = tl.program_id(0)
    channel_out_id = tl.program_id(1)
    
    # Calculate linear offset for starting position
    output_start = batch_id * C_out * H * W + channel_out_id * H * W
    
    # Load bias and scale for this output channel
    bias = tl.load(bias_ptr + channel_out_id)
    scale = tl.load(scale_ptr + channel_out_id)
    
    # Process spatial tiles
    for h_start in tl.range(0, H, BLOCK_SIZE):
        for w_start in tl.range(0, W, BLOCK_SIZE):
            # Calculate spatial tile boundaries and masks
            h_coords = h_start + tl.arange(0, BLOCK_SIZE)
            w_coords = w_start + tl.arange(0, BLOCK_SIZE)
            h_mask = h_coords < H
            w_mask = w_coords < W
            spatial_mask = h_mask[:, None] & w_mask[None, :]
            
            # Reshape for broadcasting: [BLOCK_H, BLOCK_W] -> [BLOCK_H, BLOCK_W, 1]
            spatial_mask_3d = spatial_mask[:, :, None]
            
            # Load input data for all input channels
            input_base_idx = batch_id * C_in * H * W
            # Reshape coordinates for input tensor: [H, W] -> [H, W, C_in]
            h_idx_flat = h_coords[:, None].repeat(1, BLOCK_SIZE)
            w_idx_flat = w_coords[None, :].repeat(BLOCK_SIZE, 1)
            c_idx_flat = tl.arange(0, C_in)[None, None, :].repeat(BLOCK_SIZE, BLOCK_SIZE, 1)
            
            input_idx_flat = (h_idx_flat * W + w_idx_flat).flatten() + input_base_idx
            c_idx_flat_flat = c_idx_flat.flatten()
            full_input_idx = input_idx_flat[:, None] + c_idx_flat_flat
            
            # Load input data with proper masking
            x_data = tl.load(x_ptr + full_input_idx.flatten(), mask=spatial_mask_3d.flatten(), other=0.0)
            x_data = x_data.reshape(BLOCK_SIZE, BLOCK_SIZE, C_in)
            
            # Load weight data for this specific output channel
            weight_base_idx = channel_out_id * C_in
            weight_idx = weight_base_idx + tl.arange(0, C_in)
            weights = tl.load(weight_ptr + weight_idx, mask=tl.arange(0, C_in) < C_in, other=0.0)
            weights = weights.reshape(C_in, 1, 1)  # [C_in, 1, 1] for broadcasting
            
            # Perform 1x1 convolution: sum over input channels
            conv_result = tl.sum(x_data * weights, axis=2)  # [BLOCK_H, BLOCK_W]
            
            # Add bias and apply scaling (both broadcast to spatial dimensions)
            conv_result = conv_result + bias
            conv_result = conv_result * scale
            
            # Load residual data and add
            residual_base_idx = output_start
            residual_h_idx = h_coords[:, None].repeat(1, BLOCK_SIZE)
            residual_w_idx = w_coords[None, :].repeat(BLOCK_SIZE, 1)
            residual_spatial_idx = residual_h_idx * W + residual_w_idx
            residual_idx = residual_spatial_idx.flatten() + residual_base_idx
            
            residual_data = tl.load(residual_ptr + residual_idx, mask=spatial_mask.flatten(), other=0.0)
            residual_data = residual_data.reshape(BLOCK_SIZE, BLOCK_SIZE)
            
            # Final addition and store
            final_result = conv_result + residual_data
            output_idx = residual_idx  # Reuse the same indexing pattern
            
            tl.store(out_ptr + output_idx, final_result.flatten(), mask=spatial_mask.flatten())

# Kernel wrapper function
@torch.fx.wrap
def fused_conv_scale_add(x, weight, bias, scale, residual):
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Create output tensor
    out = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)
    
    # Set block size
    BLOCK_SIZE = 32
    
    # Calculate grid dimensions
    grid = (B, C_out)
    
    # Launch Triton kernel
    fused_conv_scale_add_kernel[grid](
        x, weight, bias, scale, residual, out,
        B, C_in, C_out, H, W, BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_conv_scale_add