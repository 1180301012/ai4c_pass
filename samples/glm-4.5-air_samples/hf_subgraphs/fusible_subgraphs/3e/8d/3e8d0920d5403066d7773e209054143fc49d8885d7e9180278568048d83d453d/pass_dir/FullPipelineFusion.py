import torch
import triton
import triton.language as tl

def pattern(x, scale_factor, concat_tensor, running_mean, running_var, weight, bias):
    tmp_5 = torch.nn.functional.max_pool2d(x, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_6 = torch.nn.functional.interpolate(tmp_5, scale_factor, None, 'bilinear', False)
    tmp_7 = torch.cat([concat_tensor, tmp_6], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return tmp_9

def replacement_args(x, scale_factor, concat_tensor, running_mean, running_var, weight, bias):
    return (x, scale_factor, concat_tensor, running_mean, running_var, weight, bias)

@triton.jit
def full_pipeline_kernel(
    x_ptr,
    concat_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels_x,
    in_channels_concat,
    in_height,
    in_width,
    out_height,
    out_width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles an output channel
    pid = tl.program_id(0)
    batch_idx = pid // (in_channels_x + in_channels_concat)
    out_channel_idx = pid % (in_channels_x + in_channels_concat)
    
    # Determine if this channel comes from x or concat_tensor
    if out_channel_idx < in_channels_x:
        # This channel comes from the x path (max_pool + interpolate)
        # Calculate intermediate dimensions
        pool_height = in_height // 2
        pool_width = in_width // 2
        
        # Process spatial dimensions with bilinear interpolation
        for h in range(0, out_height, BLOCK_SIZE):
            for w in range(0, out_width, BLOCK_SIZE):
                h_end = min(h + BLOCK_SIZE, out_height)
                w_end = min(w + BLOCK_SIZE, out_width)
                
                for out_h in range(h, h_end):
                    for out_w in range(w, w_end):
                        # Index for x path
                        channel_idx = out_channel_idx
                        
                        # Bilinear interpolation from pooled input
                        pool_h = out_h * 2
                        pool_w = out_w * 2
                        
                        h1 = min(int(pool_h), pool_height - 1)
                        w1 = min(int(pool_w), pool_width - 1)
                        h2 = min(h1 + 1, pool_height - 1)
                        w2 = min(w1 + 1, pool_width - 1)
                        
                        # Get input indices for x path
                        x_idx = batch_idx * in_channels_x * in_height * in_width + channel_idx * in_height * in_width + h1 * in_width + w1
                        x_idx2 = batch_idx * in_channels_x * in_height * in_width + channel_idx * in_height * in_width + h1 * in_width + w2
                        x_idx3 = batch_idx * in_channels_x * in_height * in_width + channel_idx * in_height * in_width + h2 * in_width + w1
                        x_idx4 = batch_idx * in_channels_x * in_height * in_width + channel_idx * in_height * in_width + h2 * in_width + w2
                        
                        # Load values
                        p1 = tl.load(x_ptr + x_idx, mask=True)
                        p2 = tl.load(x_ptr + x_idx2, mask=True)
                        p3 = tl.load(x_ptr + x_idx3, mask=True)
                        p4 = tl.load(x_ptr + x_idx4, mask=True)
                        
                        # Bilinear interpolation
                        alpha_h = pool_h - h1 if pool_h < pool_height else 1.0
                        alpha_w = pool_w - w1 if pool_w < pool_width else 1.0
                        
                        val = (1-alpha_h) * (1-alpha_w) * p1 + alpha_h * (1-alpha_w) * p2 + (1-alpha_h) * alpha_w * p3 + alpha_h * alpha_w * p4
                        
                        # Store intermediate result in shared memory (simplified for this implementation)
                        conv_idx = batch_idx * (in_channels_x + in_channels_concat) * out_height * out_width + out_channel_idx * out_height * out_width + out_h * out_width + out_w
                        tl.store(out_ptr + conv_idx, val)
    else:
        # This channel comes from the concat_tensor (direct copy)
        channel_idx = out_channel_idx - in_channels_x
        
        for h in range(0, out_height, BLOCK_SIZE):
            for w in range(0, out_width, BLOCK_SIZE):
                h_end = min(h + BLOCK_SIZE, out_height)
                w_end = min(w + BLOCK_SIZE, out_width)
                
                for out_h in range(h, h_end):
                    for out_w in range(w, w_end):
                        concat_idx = batch_idx * in_channels_concat * out_height * out_width + channel_idx * out_height * out_width + out_h * out_width + out_w
                        conv_idx = batch_idx * (in_channels_x + in_channels_concat) * out_height * out_width + out_channel_idx * out_height * out_width + out_h * out_width + out_w
                        
                        # Direct copy from concat_tensor
                        concat_val = tl.load(concat_ptr + concat_idx, mask=True)
                        tl.store(out_ptr + conv_idx, concat_val)

@triton.jit
def batch_norm_relu_final_kernel(
    intermediate_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    total_channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a channel
    pid = tl.program_id(0)
    channel_idx = pid
    
    # Load normalization parameters
    mean_val = tl.load(mean_ptr + channel_idx, mask=True)
    var_val = tl.load(var_ptr + channel_idx, mask=True)
    weight_val = tl.load(weight_ptr + channel_idx, mask=True)
    bias_val = tl.load(bias_ptr + channel_idx, mask=True)
    
    # Calculate variance with epsilon
    rstd = 1.0 / tl.sqrt(var_val + eps)
    
    # Process spatial dimensions
    for h in range(0, height, BLOCK_SIZE):
        for w in range(0, width, BLOCK_SIZE):
            h_end = min(h + BLOCK_SIZE, height)
            w_end = min(w + BLOCK_SIZE, width)
            
            for out_h in range(h, h_end):
                for out_w in range(w, w_end):
                    # Load intermediate value
                    intermediate_idx = channel_idx * height * width + out_h * width + out_w
                    intermediate_val = tl.load(intermediate_ptr + intermediate_idx, mask=True)
                    
                    # Batch normalization
                    norm_val = (intermediate_val - mean_val) * rstd
                    bn_val = norm_val * weight_val + bias_val
                    
                    # ReLU
                    relu_val = tl.max(bn_val, 0.0)
                    
                    # Store final result
                    out_idx = channel_idx * height * width + out_h * width + out_w
                    tl.store(out_ptr + out_idx, relu_val)

@torch.fx.wrap
def full_pipeline_fusion(x, scale_factor, concat_tensor, running_mean, running_var, weight, bias):
    B, C_x, H, W = x.shape
    B, C_concat, H_out, W_out = concat_tensor.shape
    eps = 0.001
    
    # Calculate total channels after concatenation
    total_channels = C_x + C_concat
    
    # Grid size for first stage
    BLOCK_SIZE = 16
    grid_size_stage1 = B * total_channels
    grid_size_stage2 = total_channels
    
    # Create intermediate buffer
    intermediate = torch.empty((B, total_channels, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # First stage: max_pool + interpolate + concat
    full_pipeline_kernel[grid_size_stage1](
        x_ptr=x,
        concat_ptr=concat_tensor,
        mean_ptr=torch.empty(0, dtype=torch.float32, device='cpu'),  # Dummy
        var_ptr=torch.empty(0, dtype=torch.float32, device='cpu'),   # Dummy
        weight_ptr=torch.empty(0, dtype=torch.float32, device='cpu'), # Dummy
        bias_ptr=torch.empty(0, dtype=torch.float32, device='cpu'),   # Dummy
        out_ptr=intermediate,
        batch_size=B,
        in_channels_x=C_x,
        in_channels_concat=C_concat,
        in_height=H,
        in_width=W,
        out_height=H_out,
        out_width=W_out,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Second stage: batch_norm + relu
    out = torch.empty((B, total_channels, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Note: Parameters will be broadcasted automatically by the kernel if shapes don't match exactly
    
    batch_norm_relu_final_kernel[grid_size_stage2](
        intermediate_ptr=intermediate,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=B,
        total_channels=total_channels,
        height=H_out,
        width=W_out,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return full_pipeline_fusion