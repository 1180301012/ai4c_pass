import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, scale, residual):
    # conv2d
    conv = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # dropout (this is in the graph even though it's a no-op)
    dropped = torch.nn.functional.dropout(conv, 0.0, False, False)
    # multiply with scale (broadcasted)
    scaled = dropped * scale
    # add residual
    result = residual + scaled
    return scaled, result

def replacement_args(x, weight, bias, scale, residual):
    return (x, weight, bias, scale, residual)

@triton.jit
def fused_conv_multiply_add_kernel(
    x_ptr, weight_ptr, bias_ptr, scale_ptr, residual_ptr, out_ptr, result_ptr,
    batch, cin, cout, h, w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total_elements = batch * cout * h * w
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    for offset in offsets[mask]:
        # Map linear offset to 4D coordinates
        batch_idx = offset // (cout * h * w)
        remaining = offset % (cout * h * w)
        cout_idx = remaining // (h * w)
        spatial_idx = remaining % (h * w)
        h_idx = spatial_idx // w
        w_idx = spatial_idx % w
        
        # Load bias for this output channel
        bias_val = tl.load(bias_ptr + cout_idx, other=0.0)
        
        # Load scale value
        scale_val = tl.load(scale_ptr, other=1.0)
        
        # For 1x1 convolution: process each input channel and sum
        conv_val = 0.0
        for c_in in range(cin):
            # Load input value at this spatial position 
            in_offset = batch_idx * (cin * h * w) + c_in * (h * w) + h_idx * w + w_idx
            x_val = tl.load(x_ptr + in_offset, other=0.0)
            
            # Load weight for this output channel and input channel
            weight_offset = cout_idx * cin + c_in
            weight_val = tl.load(weight_ptr + weight_offset, other=0.0)
            
            conv_val += x_val * weight_val
        
        conv_val += bias_val
        
        # Apply scaling
        scaled_val = conv_val * scale_val
        
        # Calculate output offset for residual connection
        residual_offset = batch_idx * (cout * h * w) + cout_idx * (h * w) + h_idx * w + w_idx
        residual_val = tl.load(residual_ptr + residual_offset, other=0.0)
        
        # Add residual
        result_val = residual_val + scaled_val
        
        # Store results
        tl.store(out_ptr + offset, scaled_val, other=0.0)
        tl.store(result_ptr + offset, result_val, other=0.0)

@torch.fx.wrap
def fused_conv_multiply_add(x, weight, bias, scale, residual):
    batch, cin, h, w = x.shape
    cout = weight.shape[0]
    
    # Calculate total elements and grid size
    total_elements = batch * cout * h * w
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    scaled_out = torch.empty((batch, cout, h, w), dtype=x.dtype, device=x.device)
    result_out = torch.empty((batch, cout, h, w), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_conv_multiply_add_kernel[(num_programs,)](
        x_ptr=x, weight_ptr=weight, bias_ptr=bias, scale_ptr=scale, 
        residual_ptr=residual, out_ptr=scaled_out, result_ptr=result_out,
        batch=batch, cin=cin, cout=cout, h=h, w=w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return scaled_out, result_out

def replacement_func():
    return fused_conv_multiply_add