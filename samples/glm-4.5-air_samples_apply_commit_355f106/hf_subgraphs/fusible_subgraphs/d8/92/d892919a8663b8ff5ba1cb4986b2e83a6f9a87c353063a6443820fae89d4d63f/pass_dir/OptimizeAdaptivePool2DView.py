import torch
import triton
import triton.language as tl

def pattern(x, shape1, shape2):
    # adaptive_avg_pool2d with output size 1
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    # view operation
    viewed = pooled.view(shape1, shape2)
    return pooled, viewed

def replacement_args(x, shape1, shape2):
    return (x, shape1, shape2)

@triton.jit
def optimized_pool2d_view_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out1,
    out2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # For each output element, compute the average over the entire spatial domain
    # Since we're doing adaptive_avg_pool2d(1), we compute channel-wise averages
    
    # For 2D output [batch_size, channels] or [1, channels]
    if out2 > 1:  # [128, 128] case
        if pid >= out1 * out2:
            return
        
        row = pid // out2
        col = pid % out2
        
        if row >= batch_size * out1 or col >= out2:
            return
            
        # Compute spatial average for this channel
        total = 0.0
        count = 0
        
        # Iterate over spatial dimensions
        for h in range(in_height):
            for w in range(in_width):
                # Compute input offset
                input_offset = ((row * in_channels) + col) * (in_height * in_width) + h * in_width + w
                x_val = tl.load(x_ptr + input_offset, other=0.0)
                total += x_val
                count += 1
        
        if count > 0:
            avg_val = total / count
            tl.store(out_ptr + pid, avg_val)
    else:  # [1, channels] case  
        if pid >= batch_size * out1:
            return
            
        channel = pid
        total = 0.0
        count = 0
        
        # Iterate over spatial dimensions
        for h in range(in_height):
            for w in range(in_width):
                # Reshape input to [batch_size, channels, height, width]
                input_offset = (channel * in_height + h) * in_width + w
                x_val = tl.load(x_ptr + input_offset, other=0.0)
                total += x_val
                count += 1
        
        if count > 0:
            avg_val = total / count
            tl.store(out_ptr + pid, avg_val)

@torch.fx.wrap
def optimized_adaptive_pool2d_view(x, shape1, shape2):
    if x.dim() == 4:
        batch_size, channels, height, width = x.shape
    else:
        raise ValueError("Input must be 4D tensor")
    
    if shape2 == 1:  # [1, channels] case
        output_size = batch_size * shape1
    else:  # [128, 128] case
        output_size = shape1 * shape2
    
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(output_size, dtype=x.dtype, device=x.device)
    
    optimized_pool2d_view_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=channels,
        in_height=height,
        in_width=width,
        out1=shape1,
        out2=shape2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to 2D
    final_out = out.view(shape1, shape2)
    return final_out

def replacement_func():
    return optimized_adaptive_pool2d_view