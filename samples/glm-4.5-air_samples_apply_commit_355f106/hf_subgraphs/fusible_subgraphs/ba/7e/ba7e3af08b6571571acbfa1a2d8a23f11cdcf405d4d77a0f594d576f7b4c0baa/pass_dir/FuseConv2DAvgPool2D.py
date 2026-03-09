import torch
import triton
import triton.language as tl

# Pattern matching function to identify Conv2D + AvgPool2D sequence
def pattern(weight, x):
    tmp_0 = weight
    tmp_1 = torch.conv2d(x, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2

# Extract arguments from matched pattern
def replacement_args(weight, x):
    return (weight, x)

# Custom average pooling kernel (replacing forbidden torch function)
@triton.jit
def avg_pool2d_kernel(
    input_ptr,  # input tensor [batch, channels, height, width]
    output_ptr, # output tensor [batch, channels, height//2, width//2]
    channels: tl.constexpr,
    ih: tl.constexpr,
    iw: tl.constexpr,
):
    # Each program handles one spatial location for one channel
    pid = tl.program_id(0)
    channel_idx = pid // (ih // 2 * iw // 2)
    spatial_idx = pid % (ih // 2 * iw // 2)
    
    # Calculate output coordinates
    out_h = spatial_idx // (iw // 2)
    out_w = spatial_idx % (iw // 2)
    
    # Compute average for this location and channel
    sum_val = 0.0
    count = 0
    
    # Process 2x2 window
    for dy in range(2):
        for dx in range(2):
            in_h = out_h * 2 + dy
            in_w = out_w * 2 + dx
            
            # Only process if within bounds (no padding)
            if in_h < ih and in_w < iw:
                # Calculate input offset
                in_offset = channel_idx * ih * iw + in_h * iw + in_w
                val = tl.load(input_ptr + in_offset)
                sum_val += val
                count += 1
    
    # Store average (avoid division by zero)
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store output
    out_offset = channel_idx * (ih // 2) * (iw // 2) + out_h * (iw // 2) + out_w
    tl.store(output_ptr + out_offset, avg_val)

# Optimized 1x1 convolution with actual multi-channel computation
@triton.jit
def conv1x1_optimized_kernel(
    x_ptr,  # weight tensor [out_channels, in_channels] 
    y_ptr,  # input tensor [batch*in_channels, height, width] (flattened batch and channels)
    out_ptr, # output tensor [batch*out_channels, height, width] (flattened batch and channels)
    batch: tl.constexpr,
    ic: tl.constexpr,
    oc: tl.constexpr,
    ih: tl.constexpr,
    iw: tl.constexpr,
):
    # Each program handles one output value (one batch, one channel, one spatial location)
    pid = tl.program_id(0)
    
    total_elements = batch * oc * ih * iw
    if pid >= total_elements:
        return
    
    # Decode pid into batch, channel, and spatial indices
    batch_idx = pid // (oc * ih * iw)
    remaining = pid % (oc * ih * iw)
    
    output_channel_idx = remaining // (ih * iw)
    spatial_idx = remaining % (ih * iw)
    
    h = spatial_idx // iw
    w = spatial_idx % iw
    
    # Calculate input and output offsets for flattened tensors
    input_base_offset = batch_idx * ic * ih * iw
    output_base_offset = batch_idx * oc * ih * iw
    
    # For small number of channels (common in densenet), compute dot product directly
    # This processes all input channels for one output channel at one spatial location
    conv_val = 0.0
    
    # Load input channels for this spatial location (across all channels for this batch)
    spatial_offset = input_base_offset + spatial_idx
    for channel_idx in range(ic):
        input_val = tl.load(y_ptr + spatial_offset + channel_idx * ih * iw)
        weight_offset = output_channel_idx * ic + channel_idx
        weight_val = tl.load(x_ptr + weight_offset)
        conv_val += input_val * weight_val
    
    # Store output for this output channel and spatial location
    output_offset = output_base_offset + output_channel_idx * ih * iw + spatial_idx
    tl.store(out_ptr + output_offset, conv_val)

@torch.fx.wrap
def fused_conv_avg_pool(weight, x):
    # Get tensor shapes
    batch, ic, ih, iw = x.shape
    oc, _, _, _ = weight.shape
    
    # Step 1: Optimized Conv2D with flattened tensors to improve memory locality
    conv_out = torch.empty((batch, oc, ih, iw), device=x.device, dtype=x.dtype)
    
    # Flatten input and output for better memory access pattern
    # Input: [batch, ic, ih, iw] -> [batch*ic, ih, iw]
    x_flat = x.reshape(batch * ic, ih, iw)
    # Output: [batch, oc, ih, iw] -> [batch*oc, ih, iw]
    conv_out_flat = conv_out.reshape(batch * oc, ih, iw)
    
    # Calculate grid size: each thread computes one output value
    N_conv = batch * oc * ih * iw
    
    # Choose block size
    BLOCK_SIZE = 1024  # Adjust based on performance tuning
    num_programs_conv = (N_conv + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized Conv2D kernel
    conv1x1_optimized_kernel[(num_programs_conv,)](
        x_ptr=weight,
        y_ptr=x_flat,
        out_ptr=conv_out_flat,
        batch=batch,
        ic=ic,
        oc=oc,
        ih=ih,
        iw=iw,
    )
    
    # Step 2: Custom average pooling
    oh = ih // 2
    ow = iw // 2
    pool_out = torch.empty((batch, oc, oh, ow), device=x.device, dtype=x.dtype)
    
    # Calculate grid size for AvgPool2D
    N_pool = batch * oc * oh * ow
    
    # Launch average pooling kernel
    num_programs_pool = (N_pool + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each batch separately
    for b in range(batch):
        # Extract current batch
        batch_input = conv_out[b]  # shape: [oc, ih, iw]
        batch_output = pool_out[b]  # shape: [oc, oh, ow]
        
        # Flatten to 1D for kernel input: [oc, ih, iw] -> [oc * ih * iw]
        flat_input = batch_input.reshape(-1)
        flat_output = batch_output.reshape(-1)
        
        # Launch kernel for this batch
        avg_pool2d_kernel[(num_programs_pool,)](
            input_ptr=flat_input,
            output_ptr=flat_output,
            channels=oc,
            ih=ih,
            iw=iw,
        )
    
    return pool_out

def replacement_func():
    return fused_conv_avg_pool