import torch
import triton
import triton.language as tl

# Pattern for unnecessary view operation after convolution
def pattern(conv_input, conv_weight):
    # Match the exact operations from the model that we want to replace
    conv_output = torch.conv2d(input=conv_input, weight=conv_weight, groups=512)
    view_output = conv_output.view(1, 512, 64, 64)  # Match the view operation
    return view_output  # Return what should replace the view operation

def replacement_args(conv_input, conv_weight):
    # We only need conv_input and conv_weight args
    return (conv_input, conv_weight)



# Simple depthwise convolution kernel
@triton.jit
def depthwise_conv_kernel(
    x_ptr, 
    weight_ptr, 
    out_ptr,
    B, C, H_in, W_in, H_out, W_out,
    KH, KW,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Compute output position range for this thread
    start_h = pid_h * BLOCK_SIZE
    start_w = pid_w * BLOCK_SIZE
    
    # Process a block of output pixels
    for oh in range(BLOCK_SIZE):
        for ow in range(BLOCK_SIZE):
            h_out = start_h + oh
            w_out = start_w + ow
            
            # Only process if within bounds
            if h_out < H_out and w_out < W_out:
                # Initialize convolution result
                result = 0.0
                
                # Apply convolution
                for kh in range(KH):
                    for kw in range(KW):
                        # Calculate input position
                        h_in = h_out * stride - padding + kh * dilation
                        w_in = w_out * stride - padding + kw * dilation
                        
                        # Check bounds and accumulate
                        h_in_valid = (h_in >= 0) & (h_in < H_in)
                        w_in_valid = (w_in >= 0) & (w_in < W_in)
                        if h_in_valid and w_in_valid:
                            # Load input value
                            x_offset = (0, pid_c, h_in, w_in)
                            x_val = tl.load(x_ptr + x_offset, eviction_policy='evict_last')
                            
                            # Load weight value
                            w_offset = (pid_c, 0, kh, kw)
                            w_val = tl.load(weight_ptr + w_offset, eviction_policy='evict_last')
                            
                            result += x_val * w_val
                
                # Store result
                out_offset = (0, pid_c, h_out, w_out)
                tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def optimized_depthwise_conv(conv_input, conv_weight):
    # Get input dimensions
    B, C, H_in, W_in = conv_input.shape
    KH, KW = conv_weight.shape[2], conv_weight.shape[3]
    
    # Calculate output dimensions (assuming stride=1, padding=0 for simplicity)
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    
    # Create output tensor
    output = torch.empty((B, C, H_out, W_out), dtype=torch.float32, device=conv_input.device)
    
    # Choose block size
    BLOCK_SIZE = 8
    
    # Calculate grid dimensions
    grid_h = (H_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_w = (W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_c = C
    
    # Launch kernel
    depthwise_conv_kernel[(grid_h, grid_w, grid_c)](
        conv_input, conv_weight, output,
        B, C, H_in, W_in, H_out, W_out,
        KH, KW,
        stride=1, padding=0, dilation=1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_depthwise_conv