import torch
import triton
import triton.language as tl

# Pattern for Branch 2: sigmoid + multiply + interpolate
def pattern(conv2d_result, in_2):
    tmp_6 = torch.sigmoid(conv2d_result)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    return (tmp_6, tmp_7, tmp_8)  # Return all intermediates for proper dataflow

def replacement_args(conv2d_result, in_2):
    return (conv2d_result, in_2)

@triton.jit
def fused_sigmoid_multiply_interp_kernel(
    input_ptr,      # [B, C, H, W] - input from conv2d (16x16)
    multiplier_ptr,  # [B, C, H, W] - multiplier tensor (16x16)
    output_ptr,      # [B, C, 64, 64] - final output after interpolation
    B, C,
    input_h, input_w,   # Input spatial dimensions (16, 16)
    output_h, output_w, # Output spatial dimensions (64, 64)
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program indices
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Base pointers for this batch and channel
    input_base = batch_id * C * input_h * input_w + channel_id * input_h * input_w
    multiplier_base = batch_id * C * input_h * input_w + channel_id * input_h * input_w
    output_base = batch_id * C * output_h * output_w + channel_id * output_h * output_w
    
    # Each program processes one spatial location at output resolution
    out_h = tl.program_id(2) * BLOCK_SIZE
    out_w_offset = tl.arange(0, BLOCK_SIZE)
    
    for out_w in out_w_offset:
        if out_h >= output_h or out_w >= output_w:
            continue
            
        # Calculate source coordinates for bilinear interpolation
        src_h = (out_h * input_h) // output_h
        src_w = (out_w * input_w) // output_w
        
        # Ensure source coordinates are within bounds
        src_h = tl.maximum(0, tl.minimum(src_h, input_h - 1))
        src_w = tl.maximum(0, tl.minimum(src_w, input_w - 1))
        
        # Load input value
        input_offset = input_base + src_h * input_w + src_w
        input_val = tl.load(input_ptr + input_offset)
        
        # Apply sigmoid
        sigmoid_val = 1.0 / (1.0 + tl.exp(-input_val))
        
        # Load multiplier value at same location
        multiplier_offset = multiplier_base + src_h * input_w + src_w
        multiplier_val = tl.load(multiplier_ptr + multiplier_offset)
        
        # Apply multiplication
        mul_val = sigmoid_val * multiplier_val
        
        # Handle bilinear interpolation properly
        # Calculate exact floating point coordinates
        exact_h = out_h * input_h / output_h
        exact_w = out_w * input_w / output_w
        
        # Get integer parts and fractional parts
        h0 = tl.floor(exact_h).to(tl.int32)
        w0 = tl.floor(exact_w).to(tl.int32)
        h1 = tl.minimum(h0 + 1, input_h - 1)
        w1 = tl.minimum(w0 + 1, input_w - 1)
        
        h_frac = exact_h - h0
        w_frac = exact_w - w0
        
        # Load four corner values
        val_00 = tl.load(input_ptr + input_base + h0 * input_w + w0)
        val_01 = tl.load(input_ptr + input_base + h0 * input_w + w1)
        val_10 = tl.load(input_ptr + input_base + h1 * input_w + w0)
        val_11 = tl.load(input_ptr + input_base + h1 * input_w + w1)
        
        # Apply sigmoid to all four corners
        sigmoid_00 = 1.0 / (1.0 + tl.exp(-val_00))
        sigmoid_01 = 1.0 / (1.0 + tl.exp(-val_01))
        sigmoid_10 = 1.0 / (1.0 + tl.exp(-val_10))
        sigmoid_11 = 1.0 / (1.0 + tl.exp(-val_11))
        
        # Load four multiplier values
        mul_00 = tl.load(multiplier_ptr + multiplier_base + h0 * input_w + w0)
        mul_01 = tl.load(multiplier_ptr + multiplier_base + h0 * input_w + w1)
        mul_10 = tl.load(multiplier_ptr + multiplier_base + h1 * input_w + w0)
        mul_11 = tl.load(multiplier_ptr + multiplier_base + h1 * input_w + w1)
        
        # Multiply with sigmoid
        pm_00 = sigmoid_00 * mul_00
        pm_01 = sigmoid_01 * mul_01
        pm_10 = sigmoid_10 * mul_10
        pm_11 = sigmoid_11 * mul_11
        
        # Bilinear interpolation
        top = pm_00 * (1.0 - w_frac) + pm_01 * w_frac
        bottom = pm_10 * (1.0 - w_frac) + pm_11 * w_frac
        result = top * (1.0 - h_frac) + bottom * h_frac
        
        # Store interpolated result
        output_offset = output_base + out_h * output_w + out_w
        tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_sigmoid_multiply_interpolate_branch2(conv2d_result, in_2):
    # Get tensor dimensions
    B, C, H, W = conv2d_result.shape
    output_h, output_w = 64, 64
    
    # Create output tensor
    output = torch.empty((B, C, output_h, output_w), dtype=conv2d_result.dtype, device=conv2d_result.device)
    
    # Set block size and grid dimensions
    BLOCK_SIZE = 16  # Process 16x1 blocks per program for better memory coalescing
    grid = (B, C, (output_h * output_w + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the fused kernel
    fused_sigmoid_multiply_interp_kernel[grid](
        conv2d_result, in_2, output,
        B, C, H, W, output_h, output_w,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_sigmoid_multiply_interpolate_branch2