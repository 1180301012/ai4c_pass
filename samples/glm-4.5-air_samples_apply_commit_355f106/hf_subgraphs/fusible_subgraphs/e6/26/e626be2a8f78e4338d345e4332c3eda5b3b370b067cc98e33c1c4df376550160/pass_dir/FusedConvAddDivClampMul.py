import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Create temporary variables like the original computation
    tmp_0 = in_0
    tmp_1 = in_1
    # Conv2D operation
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    # Cleanup like original (but don't include in actual pattern match for optimization)
    # tmp_1 = tmp_0 = None
    # Add operation
    tmp_3 = tmp_2 + 1.0
    # Div operation  
    tmp_4 = tmp_3 / 2.0
    # Clamp operation
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    # Final multiplication
    tmp_6 = in_2 * tmp_5
    # Return the result that matches original
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_IC': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_IC': 16}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_IC': 32}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_IC': 64}, num_stages=3, num_warps=4),
], key=['batch_size', 'output_channels', 'input_channels'])
def fused_conv_kernel(
    bias_ptr,
    weight_ptr,
    x2_ptr, 
    x3_ptr,
    out_ptr,
    batch_size,
    output_channels,
    input_channels,
    x2_height,
    x2_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one output channel
    oc = pid
    
    # Boundary check
    if oc >= output_channels:
        return
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + oc)
    
    # Initialize accumulator for output channel across all batches
    total_conv_sum = 0.0
    
    # Optimized computation: vectorized input channel processing
    for ic in range(0, input_channels, BLOCK_SIZE_IC):
        # Process input channels in chunks
        ic_end = min(ic + BLOCK_SIZE_IC, input_channels)
        
        # Sum over chunk input channels and batches
        chunk_sum = 0.0
        for ic_local in range(ic, ic_end):
            # Load weight for this output and input channel
            weight = tl.load(weight_ptr + oc * input_channels + ic_local)
            
            # Vectorized batch processing
            batch_sum = 0.0
            for b in range(batch_size):
                # Load x3 value for this batch and input channel
                x3_idx = b * input_channels + ic_local
                x3_val = tl.load(x3_ptr + x3_idx)
                batch_sum += x3_val
            
            chunk_sum += weight * batch_sum
        
        total_conv_sum += chunk_sum
    
    # Apply operations: (conv_sum + bias) then add/div/clamp
    conv_result = total_conv_sum + bias
    fused_result = (conv_result + 1.0) / 2.0
    fused_result = tl.maximum(tl.minimum(fused_result, 1.0), 0.0)
    
    # Optimized spatial computation with vectorization
    # Precompute stride constants for better compiler optimization
    x2_batch_stride = output_channels * x2_height * x2_width
    x2_channel_stride = x2_height * x2_width
    
    # Process spatial positions in vectorized manner
    for h in range(x2_height):
        h_offset = h * x2_width
        for w in range(x2_width):
            w_offset = h_offset + w
            
            # Load x2 values for all batches at this spatial position
            batch_products = 0.0
            for b in range(batch_size):
                # Compute x2 index for this batch and spatial position
                x2_idx = b * x2_batch_stride + oc * x2_channel_stride + w_offset
                x2_val = tl.load(x2_ptr + x2_idx)
                
                # Accumulate products for spatial position
                batch_products += fused_result * x2_val
            
            # Store accumulated result for all batches at this spatial position
            out_idx = oc * x2_channel_stride + w_offset
            tl.store(out_ptr + out_idx, batch_products)

@torch.fx.wrap
def fused_conv_operation(bias, weight, x2, x3):
    # Get shapes
    batch_size = x3.shape[0]
    input_channels = x3.shape[1]
    output_channels = bias.shape[0]
    x2_height = x2.shape[2]
    x2_width = x2.shape[3]
    
    # Reshape x2 from [B, C, H, W] to [B, output_channels, H, W] for kernel access
    # This assumes x2 has output_channels as its channel dimension
    x2_reshaped = x2.reshape(batch_size * output_channels, x2_height, x2_width)
    
    # Reshape x3 to [B*input_channels] for the kernel
    x3_flat = x3.reshape(-1)
    
    # Determine output size - kernel now produces [B * output_channels * H * W]
    output_size = batch_size * output_channels * x2_height * x2_width
    out = torch.empty(output_size, dtype=torch.float32, device=bias.device)
    
    # Launch one program per output channel
    num_programs = output_channels
    
    # Execute kernel with autotuning
    fused_conv_kernel[(num_programs,)](
        bias_ptr=bias,
        weight_ptr=weight,
        x2_ptr=x2_reshaped,  # Pass as [B*output_channels, H, W] 
        x3_ptr=x3_flat,      # Pass as [B*input_channels]
        out_ptr=out,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        x2_height=x2_height,
        x2_width=x2_width,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_IC=1,  # Will be autotuned
    )
    
    # Reshape output to match expected format [B, C, H, W]
    out = out.reshape(batch_size, output_channels, x2_height, x2_width)
    
    return out

def replacement_func():
    return fused_conv_operation