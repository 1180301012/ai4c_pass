import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    # Branch 1: in_1 processing
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    
    # Branch 2: in_0 channel 1 processing  
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    
    # Branch 3: in_0 channel 2 processing
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    
    # Final concatenation
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    
    return tmp_11

def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Triton kernel for optimized computation with auto-tuning
@triton.heuristics({
    "BLOCK_SIZE": lambda args: 1024 if args["width"] * args["height"] >= 1024 else 256,
})
@triton.jit
def fused_computation_kernel(
    # Input tensors
    in_1, in_0,
    # Output tensor
    out,
    # Tensor shapes
    channels, height, width,
    # Scaling and bias parameters for each branch
    scale1, bias1,
    scale2, bias2, 
    scale3, bias3,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of spatial elements
    pid = tl.program_id(0)  # Program ID
    
    # Calculate spatial block start
    spatial_size = height * width
    block_start = pid * BLOCK_SIZE
    
    # Calculate global spatial indices
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size
    
    # Process each of the 3 branches
    
    # Branch 1: process in_1 (already [height, width], access directly)
    # For in_1, we need linear indexing since it's already 2D spatial tensor
    in_1_vals = tl.load(in_1 + offsets, mask=mask, other=0.0)
    out_0_vals = in_1_vals * scale1 + bias1  # This goes to output channel 0
    
    # Branch 2: process channel 1 from in_0 [channels, height, width]
    # We need to calculate the offset for channel 1 + spatial position
    # The tensor is laid out as [channel0_height*width, channel1_height*width, ...]
    channel_1_base = 1 * spatial_size  # base offset for channel 1
    channel_1_offsets = channel_1_base + offsets
    in_0_1_vals = tl.load(in_0 + channel_1_offsets, mask=mask, other=0.0)
    out_1_vals = in_0_1_vals * scale2 + bias2  # This goes to output channel 1
    
    # Branch 3: process channel 2 from in_0
    channel_2_base = 2 * spatial_size  # base offset for channel 2
    channel_2_offsets = channel_2_base + offsets
    in_0_2_vals = tl.load(in_0 + channel_2_offsets, mask=mask, other=0.0)
    out_2_vals = in_0_2_vals * scale3 + bias3  # This goes to output channel 2
    
    # Store results to output in correct concatenation order
    # Output: [out_0_vals, out_1_vals, out_2_vals] concatenated along channel dimension
    
    # Channel 0: out_0_vals goes to first output channel
    tl.store(out + 0 * spatial_size + offsets, out_0_vals, mask=mask)
    # Channel 1: out_1_vals goes to second output channel  
    tl.store(out + 1 * spatial_size + offsets, out_1_vals, mask=mask)
    # Channel 2: out_2_vals goes to third output channel
    tl.store(out + 2 * spatial_size + offsets, out_2_vals, mask=mask)

@torch.fx.wrap  
def fused_computation(in_1, in_0):
    # Get tensor shapes
    batch_size, _, height, width = in_1.shape  # in_1: [batch, 1, height, width]
    _, channels, _, _ = in_0.shape              # in_0: [batch, channels, height, width]
    
    # Scale and bias parameters
    scale1, bias1 = 0.458, -0.030000000000000027
    scale2, bias2 = 0.448, -0.08799999999999997
    scale3, bias3 = 0.45, -0.18799999999999994
    
    # Create output tensor: [batch, 3, height, width]
    output_shape = (batch_size, 3, height, width)
    out = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Process each batch separately
    for b in range(batch_size):
        # Extract data for current batch
        in_1_batch = in_1[b, 0].contiguous()      # Shape: [height, width]
        in_0_batch = in_0[b].contiguous()         # Shape: [channels, height, width]
        out_batch = out[b].contiguous()            # Shape: [3, height, width]
        
        # Number of spatial elements
        spatial_elements = height * width
        # Let auto-tuning choose optimal block size
        num_programs = (spatial_elements + 1024 - 1) // 1024  # Upper bound grid calculation
        
        # Launch kernel for this batch - pass tensor objects, let auto-tuning handle BLOCK_SIZE
        fused_computation_kernel[(num_programs,)](
            in_1=in_1_batch,                       # Shape: [height, width]
            in_0=in_0_batch,                       # Shape: [channels, height, width]
            out=out_batch,                          # Shape: [3, height, width]
            channels=channels,
            height=height,
            width=width,
            scale1=scale1, bias1=bias1,
            scale2=scale2, bias2=bias2,
            scale3=scale3, bias3=bias3,
        )
    
    return out

def replacement_func():
    return fused_computation