import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    out = torch.flatten(conv_out, 2)
    return out

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def fused_conv2d_flatten_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, channels_in, channels_out, height, width,
    BLOCK_BATCH: tl.constexpr, BLOCK_OUT: tl.constexpr, BLOCK_IN: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)
    
    batch_start = pid_batch * BLOCK_BATCH
    out_start = pid_out * BLOCK_OUT
    
    batch_offsets = batch_start + tl.arange(0, BLOCK_BATCH)
    out_offsets = out_start + tl.arange(0, BLOCK_OUT)
    
    batch_mask = batch_offsets < batch
    out_mask = out_offsets < channels_out
    
    # Load weight: [BLOCK_OUT, channels_in]
    weight = tl.load(weight_ptr + out_start * channels_in + tl.arange(0, channels_in),
                     mask=out_mask[:, None] & (tl.arange(0, channels_in) < channels_in),
                     other=0.0)
    
    # Load bias: [BLOCK_OUT]
    bias = tl.load(bias_ptr + out_start + tl.arange(0, BLOCK_OUT),
                   mask=out_mask,
                   other=0.0)
    
    # Process spatial dimension in chunks
    for spatial_idx in range(0, height * width, BLOCK_IN):
        spatial_end = min(spatial_idx + BLOCK_IN, height * width)
        num_spatial = spatial_end - spatial_idx
        
        # Calculate input indices: [BLOCK_BATCH, channels_in, num_spatial]
        input_indices = (batch_offsets[:, None, None] * channels_in * height * width +
                         tl.arange(0, channels_in)[None, :, None] * height * width +
                         spatial_idx + tl.arange(0, num_spatial)[None, None, :])
        
        # Load input data
        input_data = tl.load(input_ptr + input_indices,
                             mask=batch_mask[:, None, None] & (tl.arange(0, num_spatial)[None, None, :] < num_spatial),
                             other=0.0)
        
        # Compute dot product: (BLOCK_BATCH, num_spatial) = input_data @ weight.T
        acc = tl.dot(input_data, weight.T)
        
        # Calculate output indices
        output_indices = (batch_offsets[:, None] * channels_out * height * width +
                          out_offsets[:, None] * height * width +
                          spatial_idx + tl.arange(0, num_spatial)[None, :])
        
        # Load existing output
        output = tl.load(output_ptr + output_indices,
                         mask=batch_mask[:, None] & out_mask[:, None] & (tl.arange(0, num_spatial)[None, :] < num_spatial),
                         other=0.0)
        
        # Update output with accumulated result and bias
        output += acc
        output += bias[:, None]
        
        # Store back to output
        tl.store(output_ptr + output_indices, output,
                 mask=batch_mask[:, None] & out_mask[:, None] & (tl.arange(0, num_spatial)[None, :] < num_spatial))

@torch.fx.wrap
def fused_conv2d_flatten(x, weight, bias):
    # Reshape weight from [C_out, C_in, 1, 1] to [C_out, C_in]
    weight_2d = weight.squeeze(2).squeeze(2)
    B, C_in, H, W = x.shape
    C_out = weight_2d.shape[0]
    
    # Output shape: [B, C_out, H*W]
    output = torch.empty((B, C_out, H * W), dtype=x.dtype, device=x.device)
    
    # Configure block sizes
    BLOCK_BATCH = 16
    BLOCK_OUT = 32
    BLOCK_IN = 64
    
    grid = (triton.cdiv(B, BLOCK_BATCH), triton.cdiv(C_out, BLOCK_OUT))
    
    # Launch kernel
    fused_conv2d_flatten_kernel[grid](
        x, weight_2d, bias, output,
        B, C_in, C_out, H, W,
        BLOCK_BATCH=BLOCK_BATCH, BLOCK_OUT=BLOCK_OUT, BLOCK_IN=BLOCK_IN
    )
    return output

def replacement_func():
    return fused_conv2d_flatten