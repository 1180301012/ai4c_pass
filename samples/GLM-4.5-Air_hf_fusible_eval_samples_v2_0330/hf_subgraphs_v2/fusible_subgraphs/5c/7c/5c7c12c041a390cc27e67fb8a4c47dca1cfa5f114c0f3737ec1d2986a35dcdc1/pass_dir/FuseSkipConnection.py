import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple identity pattern for testing"""
    return x

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def identity_skip(x):
    """Simple identity operation for testing"""
    return x

def replacement_func():
    return identity_skip
    """Simple identity pattern for testing"""
    return x

def replacement_args(x):
    return (x,)

@triton.jit
def fused_skip_connection_kernel(
    x1_ptr,  # Input that needs downsampling
    x2_ptr,  # Input that stays the same
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels1,
    channels2,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one spatial location
    batch_id = pid // (height * width)
    spatial_id = pid % (height * width)
    h = spatial_id // width
    w = spatial_id % width
    
    # Bounds checking
    if batch_id >= batch_size or h >= height or w >= width:
        return
    
    # Each program handles multiple channels for better occupancy
    for c_base in range(0, channels1 + channels2, BLOCK_SIZE):
        c_end = min(c_base + BLOCK_SIZE, channels1 + channels2)
        channel_mask = c_base < (channels1 + channels2)
        
        # Determine which channel we're processing
        if c_base < channels1:
            # Processing from downscaled x1
            chan_idx = c_base
            # Load from downsampled x1 (coordinates are already half)
            load_h, load_w = h, w
            is_x2 = False
        else:
            # Processing from x2  
            chan_idx = c_base
            load_h, load_w = h, w
            is_x2 = True
        
        # Load input value
        if is_x2:
            # x2 is already at target size
            offset = (batch_id * (channels1 + channels2) + chan_idx) * height * width + h * width + w
            x_val = tl.load(x2_ptr + offset, mask=channel_mask, other=0.0)
        else:
            # x1 needs 2x2 max pooling from original coordinates
            orig_h, orig_w = h * 2, w * 2
            max_val = -float('inf')
            
            # 2x2 max pooling
            for dh in range(2):
                for dw in range(2):
                    cur_h = orig_h + dh
                    cur_w = orig_w + dw
                    
                    if cur_h < (height * 2) and cur_w < (width * 2):
                        pool_offset = (batch_id * channels1 + cur_h) * (width * 2) + cur_w
                        val = tl.load(x1_ptr + pool_offset, other=-float('inf'))
                        max_val = tl.math.maximum(max_val, val)
            
            x_val = max_val
        
        # Load batch norm parameters if this is in the batch norm range
        if chan_idx < channels1 + channels2:
            # Load parameters (channels1+channels2 total for batch norm)
            param_offset = chan_idx
            weight_val = tl.load(weight_ptr + param_offset, mask=channel_mask, other=1.0)
            bias_val = tl.load(bias_ptr + param_offset, mask=channel_mask, other=0.0)
            mean_val = tl.load(running_mean_ptr + param_offset, mask=channel_mask, other=0.0)
            var_val = tl.load(running_var_ptr + param_offset, mask=channel_mask, other=1.0)
            
            # Batch norm calculation
            var_eps = var_val + eps
            inv_std = tl.math.rsqrt(var_eps)
            normalized = (x_val - mean_val) * inv_std
            bn_val = normalized * weight_val + bias_val
            
            # ReLU activation
            relu_val = tl.math.maximum(bn_val, 0.0)
        else:
            relu_val = 0.0
        
        # Store result
        out_offset = offset  # Same as input offset
        tl.store(out_ptr + out_offset, relu_val, mask=channel_mask)

@torch.fx.wrap
def identity_skip(x):
    """Simple identity operation for testing"""
    return x

def replacement_func():
    return identity_skip
    
    # The kernel outputs only the final relu result, but we need to return both concat_out and relu_out
    # For now, return the concat_out from the original and relu_out from the kernel
    # In a more advanced implementation, we'd compute both in the kernel
    pool_out = torch.nn.functional.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
    interp_out = torch.nn.functional.interpolate(pool_out, size=(h2, w2), mode='bilinear', align_corners=False)
    concat_out = torch.cat([x2, interp_out], dim=1)
    
    return concat_out, out

def replacement_func():
    return fused_skip_connection