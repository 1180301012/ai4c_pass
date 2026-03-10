import torch
import triton
import triton.language as tl

def pattern(x, weight):
    tmp_0 = weight
    tmp_1 = torch.conv2d(x, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_0 = None
    tmp_2 = torch.nn.functional.max_pool2d(tmp_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    tmp_1 = None
    return tmp_1, tmp_2

def replacement_args(x, weight):
    return (x, weight)

@triton.jit
def fused_conv_pool_kernel(
    weight_ptr,  # [out_channels, in_channels, kernel_h, kernel_w]
    x_ptr,       # [batch_size, in_channels, height, width]
    out_ptr,     # [batch_size, out_channels, out_height, out_width]
    batch_size,
    in_channels,
    in_height, 
    in_width,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    use_bias,
    pool_kernel_size,
    pool_stride,
    pool_padding,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    pool_out_height = (out_height + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    pool_out_width = (out_width + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    
    # Program identifiers
    pid = tl.program_id(0)
    batch_idx = pid // (pool_out_height * pool_out_width)
    pool_out_h = (pid // pool_out_width) % pool_out_height
    pool_out_w = pid % pool_out_width
    
    # Calculate corresponding conv output location
    conv_out_h = pool_out_h * pool_stride - pool_padding
    conv_out_w = pool_out_w * pool_stride - pool_padding
    
    # Ensure we're within bounds
    if batch_idx >= batch_size or conv_out_h < 0 or conv_out_h >= out_height or conv_out_w < 0 or conv_out_w >= out_width:
        return
    
    # Initialize accumulator for the pooling operation
    max_val = float('-inf')
    
    # Iterate over the pooling window
    for pool_i in range(pool_kernel_size):
        for pool_j in range(pool_kernel_size):
            conv_h = conv_out_h + pool_i
            conv_w = conv_out_w + pool_j
            
            # Check bounds for conv output
            if 0 <= conv_h < out_height and 0 <= conv_w < out_width:
                # Compute convolution at this location and update max
                conv_val = compute_conv_at_position(
                    batch_idx, conv_h, conv_w,
                    weight_ptr, x_ptr,
                    batch_size, in_channels, in_height, in_width,
                    out_channels, kernel_size, stride, padding, dilation, groups
                )
                if conv_val > max_val:
                    max_val = conv_val
    
    # Store the pooled result
    tl.store(out_ptr + batch_idx * pool_out_height * pool_out_width * out_channels +
             pool_out_h * pool_out_width * out_channels + pool_out_w * out_channels, 
             max_val)

@triton.jit
def compute_conv_at_position(
    batch_idx, out_h, out_w,
    weight_ptr, x_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_size, stride, padding, dilation, groups
):
    acc = 0.0
    
    # Iterate over output channels
    for oc in range(out_channels):
        channel_offset = out_channels * out_height * out_width
        weight_base = oc * in_channels * kernel_size * kernel_size
        
        for ic in range(in_channels):
            # Handle groups
            if groups > 1 and oc // (out_channels // groups) != ic // (in_channels // groups):
                continue
                
            x_base = batch_idx * in_channels * in_height * in_width + ic * in_height * in_width
            
            # Iterate over kernel
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_h = out_h * stride - padding + kh * dilation
                    in_w = out_w * stride - padding + kw * dilation
                    
                    if 0 <= in_h < in_height and 0 <= in_w < in_width:
                        # Load weight and input
                        weight_idx = weight_base + ic * kernel_size * kernel_size + kh * kernel_size + kw
                        x_idx = x_base + in_h * in_width + in_w
                        
                        weight_val = tl.load(weight_ptr + weight_idx)
                        x_val = tl.load(x_ptr + x_idx)
                        
                        acc += weight_val * x_val
    
    return acc

@triton.jit
def simple_fused_conv_pool_kernel(
    x_ptr,       # [batch_size, in_channels, height, width]
    weight_ptr,  # [out_channels, in_channels, kernel_h, kernel_w]
    out_ptr,     # [batch_size, out_channels, out_height, out_width]
    batch_size,
    in_channels,
    height, 
    width,
    out_channels,
    conv_kernel_size,
    conv_stride,
    conv_padding,
    conv_dilation,
    pool_kernel_size,
    pool_stride,
    pool_padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (conv_kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (conv_kernel_size - 1) - 1) // conv_stride + 1
    pool_out_height = (out_height + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    pool_out_width = (out_width + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    
    # Create program identifiers using flattened grid
    pid = tl.program_id(0)
    total_tiles = (pool_out_height + BLOCK_SIZE - 1) // BLOCK_SIZE * (pool_out_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if pid >= total_tiles * batch_size:
        return
    
    # Extract batch_idx and tile position
    batch_idx = pid // total_tiles
    local_pid = pid % total_tiles
    
    # Calculate tile coordinates
    tiles_x = (pool_out_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_y = local_pid // tiles_x
    tile_x = local_pid % tiles_x
    
    # Calculate starting position for this tile in pool output
    start_pool_h = tile_y * BLOCK_SIZE
    start_pool_w = tile_x * BLOCK_SIZE
    
    # Process this tile of the pooling operation
    for pool_h in range(start_pool_h, min(start_pool_h + BLOCK_SIZE, pool_out_height)):
        for pool_w in range(start_pool_w, min(start_pool_w + BLOCK_SIZE, pool_out_width)):
            # Calculate corresponding conv output location
            conv_out_h = pool_h * pool_stride - pool_padding
            conv_out_w = pool_w * pool_stride - pool_padding
            
            if 0 <= conv_out_h < out_height and 0 <= conv_out_w < out_width:
                # Compute max over pooling window
                max_val = float('-inf')
                
                for pool_i in range(pool_kernel_size):
                    for pool_j in range(pool_kernel_size):
                        conv_h = conv_out_h + pool_i
                        conv_w = conv_out_w + pool_j
                        
                        if 0 <= conv_h < out_height and 0 <= conv_w < out_width:
                            # Compute single convolution output
                            conv_val = 0.0
                            
                            for oc in range(out_channels):
                                for ic in range(in_channels):
                                    # Load weights for this output channel and input channel
                                    weight_idx = oc * in_channels * conv_kernel_size * conv_kernel_size + ic * conv_kernel_size * conv_kernel_size
                                    
                                    for kh in range(conv_kernel_size):
                                        for kw in range(conv_kernel_size):
                                            in_h = conv_h * conv_stride - conv_padding + kh * conv_dilation
                                            in_w = conv_w * conv_stride - conv_padding + kw * conv_dilation
                                            
                                            if 0 <= in_h < height and 0 <= in_w < width:
                                                x_idx = batch_idx * in_channels * height * width + ic * height * width + in_h * width + in_w
                                                weight_val = tl.load(weight_ptr + weight_idx + kh * conv_kernel_size + kw)
                                                x_val = tl.load(x_ptr + x_idx)
                                                conv_val += weight_val * x_val
                            
                            if conv_val > max_val:
                                max_val = conv_val
                
                # Store the result
                out_idx = batch_idx * out_channels * pool_out_height * pool_out_width + pool_h * pool_out_width + pool_w
                tl.store(out_ptr + out_idx, max_val)

@triton.jit
def simple_fused_kernel(
    x_ptr,          # [batch_size, in_channels, height, width]
    weight_ptr,     # [out_channels, in_channels, kernel_h, kernel_w]
    conv_out_ptr,   # [batch_size, out_channels, conv_out_h, conv_out_w]
    final_out_ptr,  # [batch_size, out_channels, pool_out_h, pool_out_w]
    batch_size,
    in_channels,
    height, 
    width,
    out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output dimensions
    conv_out_h = height  # stride=1, padding=1, dilation=1, kernel_size=1x1
    conv_out_w = width
    pool_out_h = (conv_out_h + 2 * 1 - 3) // 2 + 1  # stride=2, padding=1, kernel=3x3
    pool_out_w = (conv_out_w + 2 * 1 - 3) // 2 + 1
    
    # Program identifiers
    pid = tl.program_id(0)
    batch_idx = pid // (pool_out_h * pool_out_w)
    pool_h = (pid // pool_out_w) % pool_out_h
    pool_w = pid % pool_out_w
    
    # Calculate conv output location for pooling
    conv_h = pool_h * 2 - 1  # stride=2, from pool
    conv_w = pool_w * 2 - 1
    
    # Process each output channel
    for oc in tl.range(0, out_channels, BLOCK_SIZE):
        channel_block = min(BLOCK_SIZE, out_channels - oc)
        
        # Store intermediate conv result
        if 0 <= conv_h < conv_out_h and 0 <= conv_w < conv_out_w:
            conv_idx = (batch_idx * out_channels + oc) * conv_out_h * conv_out_w + conv_h * conv_out_w + conv_w
            
            # Compute simple 1x1 convolution
            conv_val = 0.0
            for ic in range(in_channels):
                weight_idx = (oc + 0) * in_channels * 1 * 1 + ic * 1 * 1 + 0 * 1 + 0
                x_idx = (batch_idx * in_channels + ic) * height * width + conv_h * width + conv_w
                
                conv_val += tl.load(weight_ptr + weight_idx) * tl.load(x_ptr + x_idx)
            
            tl.store(conv_out_ptr + conv_idx, conv_val)
        
        # Compute max pooling over 3x3 window
        max_val = float('-inf')
        for pool_i in range(3):
            for pool_j in range(3):
                check_h = conv_h + pool_i - 1
                check_w = conv_w + pool_j - 1
                
                if 0 <= check_h < conv_out_h and 0 <= check_w < conv_out_w:
                    # Compute conv at this position
                    pool_conv_val = 0.0
                    for ic in range(in_channels):
                        weight_idx = (oc + 0) * in_channels * 1 * 1 + ic * 1 * 1 + 0 * 1 + 0
                        x_idx = (batch_idx * in_channels + ic) * height * width + check_h * width + check_w
                        
                        pool_conv_val += tl.load(weight_ptr + weight_idx) * tl.load(x_ptr + x_idx)
                    
                    if pool_conv_val > max_val:
                        max_val = pool_conv_val
        
        # Store pooled result
        pool_idx = (batch_idx * out_channels + oc) * pool_out_h * pool_out_w + pool_h * pool_out_w + pool_w
        tl.store(final_out_ptr + pool_idx, max_val)

@torch.fx.wrap
def fused_conv2d_maxpool2d(x, weight):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = weight.shape
    
    # Calculate output dimensions
    conv_out_h = height
    conv_out_w = width
    pool_out_h = (conv_out_h + 2 * 1 - 3) // 2 + 1
    pool_out_w = (conv_out_w + 2 * 1 - 3) // 2 + 1
    
    # Create output tensors
    conv_out = torch.empty(batch_size, out_channels, conv_out_h, conv_out_w, device=x.device, dtype=x.dtype)
    final_out = torch.empty(batch_size, out_channels, pool_out_h, pool_out_w, device=x.device, dtype=x.dtype)
    
    # Launch kernel
    total_elements = batch_size * pool_out_h * pool_out_w
    grid_size = (total_elements + 1023) // 1024
    
    simple_fused_kernel[grid_size](
        x_ptr=x,
        weight_ptr=weight,
        conv_out_ptr=conv_out,
        final_out_ptr=final_out,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        out_channels=out_channels,
        BLOCK_SIZE=64
    )
    
    return conv_out, final_out

def replacement_func():
    return fused_conv2d_maxpool2d