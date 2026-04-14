import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly mirror the model.py computation
def pattern(x):
    # ReLU activation with inplace=True
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    # Three identical max_pool2d operations
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    # Concatenate along channel dimension (dim=1)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    # Return the observable outputs - must match the model's return structure
    return tmp_4

# Extract arguments for the replacement function
def replacement_args(x):
    return (x,)

# Fused Triton kernel for ReLU + Triple MaxPool2D + Concatenation
@triton.jit
def fused_relu_triple_maxpool_kernel(
    x_ptr,                    # Input tensor after ReLU (already computed)
    relu_out_ptr,            # ReLU output (original input to max_pool)
    max_pool_out1_ptr,       # First max_pool output
    max_pool_out2_ptr,       # Second max_pool output  
    max_pool_out3_ptr,       # Third max_pool output
    concat_out_ptr,          # Final concatenated output
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    
    # Tensor strides
    x_stride_batch,
    x_stride_channel,
    x_stride_height,
    x_stride_width,
    
    relu_out_stride_batch,
    relu_out_stride_channel,
    relu_out_stride_height,
    relu_out_stride_width,
    
    max_pool_stride_batch,
    max_pool_stride_channel,
    max_pool_stride_height,
    max_pool_stride_width,
    
    concat_stride_batch,
    concat_stride_channel,
    concat_stride_height,
    concat_stride_width,
    
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for channel dimension  
    BLOCK_SIZE_H: tl.constexpr,  # Block size for height dimension
):
    # Program identifiers
    m = tl.program_id(0)  # batch dimension
    n = tl.program_id(1)  # channel dimension groups
    
    # Compute channel range for this program
    channel_start = n * BLOCK_SIZE_N
    channel_end = min(channel_start + BLOCK_SIZE_N, in_channels)
    
    # Max pooling parameters (5x5 kernel, stride 1, padding 2)
    kernel_size = 5
    stride = 1
    padding = 2
    
    # Process each batch
    for b in range(0, batch_size, BLOCK_SIZE_M):
        batch_idx = b + tl.arange(0, BLOCK_SIZE_M)
        batch_mask = batch_idx < batch_size
        
        # Process each channel group
        for c in range(channel_start, channel_end, BLOCK_SIZE_N):
            channel_range = tl.arange(0, BLOCK_SIZE_N)
            channel_mask = c + channel_range < in_channels
            
            # For each spatial location, process with block tiling
            # Compute offsets within the input spatial dimensions
            h_start = tl.arange(0, BLOCK_SIZE_H)
            w_start = tl.arange(0, BLOCK_SIZE_H)
            
            # Process spatial locations with tiling
            for h_out in range(0, out_height, BLOCK_SIZE_H):
                w_out = tl.arange(0, min(BLOCK_SIZE_H, out_width - h_out))
                h_out_vec = h_out + w_out
                
                # Compute input spatial coordinates considering padding
                h_in_start = h_out_vec * stride
                w_in_start = tl.arange(0, min(BLOCK_SIZE_H, out_width - h_out))
                w_in_vec = w_in_start * stride
                
                # Load input window for max pooling (5x5)
                # Note: This is simplified - in practice we'd need to handle the convolution window
                # For now, we'll load the 5x5 window around each output location
                values = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
                
                # Process each output location in the block
                for i in range(BLOCK_SIZE_H):
                    for j in range(BLOCK_SIZE_H):
                        # Compute input coordinates with padding
                        h_in_base = h_in_start + i if tl.static_size(h_out_vec) > i else 0
                        w_in_base = w_in_vec[j] if j < len(w_in_vec) else 0
                        
                        # Load and compute max over 5x5 window
                        max_val = tl.float32('-inf')
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                h_in = h_in_base + kh - padding
                                w_in = w_in_base + kw - padding
                                
                                # Skip if outside valid input range
                                if (h_in >= 0 and h_in < in_height and 
                                    w_in >= 0 and w_in < in_width and 
                                    b + tl.arange(0, BLOCK_SIZE_M)[i % BLOCK_SIZE_M] < batch_size and 
                                    c + channel_range[0] < in_channels):
                                    
                                    # Load input value
                                    offset_batch = (b + (i % BLOCK_SIZE_M)) * x_stride_batch
                                    offset_channel = c * x_stride_channel
                                    offset_height = h_in * x_stride_height
                                    offset_width = w_in * x_stride_width
                                    
                                    val = tl.load(x_ptr + offset_batch + offset_channel + offset_height + offset_width, 
                                                mask=batch_mask[(i % BLOCK_SIZE_M)],
                                                other=tl.float32('-inf'))
                                    max_val = tl.maximum(max_val, val)
                        
                        # Store max_pool values in three output tensors
                        batch_offset = (b + (i % BLOCK_SIZE_M)) * max_pool_stride_batch
                        channel_offset1 = c * max_pool_stride_channel
                        channel_offset2 = (in_channels + c) * max_pool_stride_channel  
                        channel_offset3 = (2 * in_channels + c) * max_pool_stride_channel
                        
                        height_offset = (h_out + i) * max_pool_stride_height if i < len(h_out_vec) else 0
                        width_offset = (h_out + i) * max_pool_stride_width if i < len(h_out_vec) else 0
                        
                        # Store three identical max_pool results
                        ptr1 = max_pool_out1_ptr + batch_offset + channel_offset1 + height_offset + width_offset
                        ptr2 = max_pool_out2_ptr + batch_offset + channel_offset2 + height_offset + width_offset  
                        ptr3 = max_pool_out3_ptr + batch_offset + channel_offset3 + height_offset + width_offset
                        
                        if i < len(h_out_vec):
                            tl.store(ptr1, max_val, mask=batch_mask[(i % BLOCK_SIZE_M)])
                            tl.store(ptr2, max_val, mask=batch_mask[(i % BLOCK_SIZE_M)])  
                            tl.store(ptr3, max_val, mask=batch_mask[(i % BLOCK_SIZE_M)])
                
                # Also copy the original ReLU output to first output channel group
                for i in range(BLOCK_SIZE_H):
                    for j in range(BLOCK_SIZE_H):
                        batch_offset = (b + (i % BLOCK_SIZE_M)) * relu_out_stride_batch
                        channel_offset = c * relu_out_stride_channel
                        height_offset = (h_out + i) * relu_out_stride_height if i < len(h_out_vec) else 0
                        width_offset = (h_out + i) * relu_out_stride_width if i < len(h_out_vec) else 0
                        
                        src_ptr = x_ptr + batch_offset + channel_offset + height_offset + width_offset
                        dst_ptr = relu_out_ptr + batch_offset + channel_offset + height_offset + width_offset
                        
                        if i < len(h_out_vec):
                            val = tl.load(src_ptr, mask=batch_mask[(i % BLOCK_SIZE_M)], other=0.0)
                            tl.store(dst_ptr, val, mask=batch_mask[(i % BLOCK_SIZE_M)])
                
                # Concatenate all four outputs
                for i in range(BLOCK_SIZE_H):
                    for j in range(BLOCK_SIZE_H):
                        batch_offset = (b + (i % BLOCK_SIZE_M)) * concat_stride_batch
                        channel_offset1 = c * concat_stride_channel
                        channel_offset2 = (in_channels + c) * concat_stride_channel
                        channel_offset3 = (2 * in_channels + c) * concat_stride_channel
                        channel_offset4 = (3 * in_channels + c) * concat_stride_channel
                        
                        height_offset = (h_out + i) * concat_stride_height if i < len(h_out_vec) else 0
                        width_offset = (h_out + i) * concat_stride_width if i < len(h_out_vec) else 0
                        
                        # Load four components
                        src1 = relu_out_ptr + batch_offset + channel_offset1 + height_offset + width_offset
                        src2 = max_pool_out1_ptr + batch_offset + channel_offset2 + height_offset + width_offset
                        src3 = max_pool_out2_ptr + batch_offset + channel_offset3 + height_offset + width_offset  
                        src4 = max_pool_out3_ptr + batch_offset + channel_offset4 + height_offset + width_offset
                        
                        dst = concat_out_ptr + batch_offset + (4 * c) * concat_stride_channel + height_offset + width_offset
                        
                        if i < len(h_out_vec):
                            val1 = tl.load(src1, mask=batch_mask[(i % BLOCK_SIZE_M)], other=0.0)
                            val2 = tl.load(src2, mask=batch_mask[(i % BLOCK_SIZE_M)], other=0.0)
                            val3 = tl.load(src3, mask=batch_mask[(i % BLOCK_SIZE_M)], other=0.0)  
                            val4 = tl.load(src4, mask=batch_mask[(i % BLOCK_SIZE_M)], other=0.0)
                            
                            # Store in concatenation order: [tmp_0, tmp_1, tmp_2, tmp_3]
                            tl.store(dst + 0, val1, mask=batch_mask[(i % BLOCK_SIZE_M)])
                            tl.store(dst + 1, val2, mask=batch_mask[(i % BLOCK_SIZE_M)])
                            tl.store(dst + 2, val3, mask=batch_mask[(i % BLOCK_SIZE_M)])
                            tl.store(dst + 3, val4, mask=batch_mask[(i % BLOCK_SIZE_M)])

# Wrapper function decorated with @torch.fx.wrap
@torch.fx.wrap  
def fused_relu_triple_maxpool(x):
    # Get tensor metadata
    batch_size, in_channels, in_height, in_width = x.shape
    out_height = in_height + 2 * 2 - 5  # With padding 2, kernel 5, stride 1
    out_width = in_width + 2 * 2 - 5
    
    # Calculate output shapes
    relu_out_shape = (batch_size, in_channels, in_height, in_width)
    max_pool_shape = (batch_size, in_channels, out_height, out_width)
    concat_shape = (batch_size, 4 * in_channels, out_height, out_width)
    
    # Allocate output tensors
    relu_out = torch.empty_like(x)
    max_pool_out1 = torch.empty(batch_size, in_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    max_pool_out2 = torch.empty(batch_size, in_channels, out_height, out_width, dtype=x.dtype, device=x.device) 
    max_pool_out3 = torch.empty(batch_size, in_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    concat_out = torch.empty(batch_size, 4 * in_channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Triton kernel configuration - adapt to typical input sizes
    # Use larger blocks for better GPU utilization with moderate batch sizes
    BLOCK_SIZE_M = 1 if batch_size <= 8 else 2  
    BLOCK_SIZE_N = 64  # Process 64 channels at a time
    BLOCK_SIZE_H = 8   # Process 8x8 spatial blocks at a time
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (in_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    # Launch kernel with 3D grid (batch, channel, spatial)
    fused_relu_triple_maxpool_kernel[(grid_m, grid_n, grid_h)](
        x, relu_out, max_pool_out1, max_pool_out2, max_pool_out3, concat_out,
        batch_size, in_channels, in_height, in_width, out_height, out_width,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        relu_out.stride(0), relu_out.stride(1), relu_out.stride(2), relu_out.stride(3),
        max_pool_out1.stride(0), max_pool_out1.stride(1), max_pool_out1.stride(2), max_pool_out1.stride(3),
        max_pool_out2.stride(0), max_pool_out2.stride(1), max_pool_out2.stride(2), max_pool_out2.stride(3),
        max_pool_out3.stride(0), max_pool_out3.stride(1), max_pool_out3.stride(2), max_pool_out3.stride(3),
        concat_out.stride(0), concat_out.stride(1), concat_out.stride(2), concat_out.stride(3),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_H
    )
    
    return concat_out

# Replacement function - returns function reference (no arguments)
def replacement_func():
    return fused_relu_triple_maxpool