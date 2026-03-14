import torch
import triton
import triton.language as tl


def pattern(input_in):
    tmp_5 = torch.nn.functional.max_pool2d(input_in, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_6 = torch.nn.functional.interpolate(tmp_5, (256, 256), None, 'bilinear', False)
    return tmp_6


def replacement_args(input_in):
    return (input_in,)


@triton.jit
def fused_pool_interpolate_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Calculate grid coordinates
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Each block processes BLOCK_SIZE_M x BLOCK_SIZE_N pixels
    output_y_base = m * BLOCK_SIZE_M
    output_x_base = n * BLOCK_SIZE_N
    
    # Calculate input coordinates (since we're doing 2x2 max pool then interpolate)
    # The interpolate operation works between pooled coordinates
    for ky in range(0, BLOCK_SIZE_M):
        for kx in range(0, BLOCK_SIZE_N):
            output_y = output_y_base + ky
            output_x = output_x_base + kx
            
            if output_y < output_height and output_x < output_width:
                # This maps to interpolated input coordinates
                # Since we used bilinear interpolation, we need to sample around the 4-nearest neighbors
                # after max pooling
                # First: map to max_pool coordinates (2x2 downsample)
                pool_y = output_y * 2
                pool_x = output_x * 2
                
                # Bilinear interpolation weights
                y_frac = (output_y * 2) - pool_y
                x_frac = (output_x * 2) - pool_x
                
                # Sample 2x2 region after max pooling (since max_pool reduces by factor of 2)
                max_pool_height = input_height // 2
                max_pool_width = input_width // 2
                
                # Load 2x2 region for interpolation
                values = []
                weights = []
                
                # Sample 4 bilinear interpolation points
                for dy in range(2):
                    for dx in range(2):
                        sample_y = pool_y + dy
                        sample_x = pool_x + dx
                        
                        if sample_y < max_pool_height and sample_x < max_pool_width:
                            # Load max pool value
                            offset = sample_y * max_pool_width * n_channels + sample_x * n_channels
                            val = tl.load(input_ptr + offset)
                            values.append(val)
                            # Bilinear weights
                            weight = (1 - abs((sample_y - pool_y) - y_frac)) * (1 - abs((sample_x - pool_x) - x_frac))
                            weights.append(weight)
                
                # Perform weighted interpolation
                result = tl.zeros([n_channels], dtype=tl.float32)
                total_weight = 0.0
                for i in range(len(values)):
                    if i < len(weights):
                        for c in range(n_channels):
                            result[c] += values[i][c] * weights[i]
                        total_weight += weights[i]
                
                if total_weight > 0:
                    for c in range(n_channels):
                        result[c] /= total_weight
                
                # Store result
                output_offset = output_y * output_width * n_channels + output_x * n_channels
                tl.store(output_ptr + output_offset, result)


@torch.fx.wrap
def fused_pool_interpolate(input_in, interpolate_size):
    input_shape = input_in.shape
    n_channels, input_height, input_width = input_shape[0], input_shape[2], input_shape[3]
    output_height, output_width = interpolate_size
    
    # Create output tensor
    output_shape = [n_channels, output_height, output_width]
    output = torch.empty(output_shape, dtype=input_in.dtype, device=input_in.device)
    
    # Configure block sizes
    BLOCK_SIZE_M = 16 if output_height >= 16 else output_height
    BLOCK_SIZE_N = 16 if output_width >= 16 else output_width
    BLOCK_SIZE_K = n_channels
    
    # Calculate grid size
    grid_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    fused_pool_interpolate_kernel[grid](
        input_ptr=input_in,
        output_ptr=output,
        n_channels=n_channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


def replacement_func():
    return fused_pool_interpolate