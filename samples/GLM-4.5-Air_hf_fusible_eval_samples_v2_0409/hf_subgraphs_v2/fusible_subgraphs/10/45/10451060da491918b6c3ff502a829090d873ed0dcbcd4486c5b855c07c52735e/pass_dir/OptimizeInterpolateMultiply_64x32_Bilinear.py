import torch
import triton
import triton.language as tl

def pattern(tmp_2, in_2):
    """Match Interpolate followed by Element-wise multiplication"""
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4

def replacement_args(tmp_2, in_2):
    """Extract input tensors for the replacement"""
    return (tmp_2, in_2)

@triton.jit
def optimized_interpolate_multiply_kernel(
    input_ptr, multiply_ptr, out_ptr,
    batch, channels, in_height, in_width, out_height, out_width,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    """Optimized kernel for bilinear interpolation + multiplication"""
    # Each program handles a tile of the output tensor  
    pid_c = tl.program_id(0)  # Channel block
    pid_h = tl.program_id(1)  # Height block  
    pid_w = tl.program_id(2)  # Width block
    
    # Calculate ranges for this program
    start_c = pid_c * BLOCK_SIZE_C
    start_h = pid_h * BLOCK_SIZE_H  
    start_w = pid_w * BLOCK_SIZE_W
    
    end_c = tl.minimum(start_c + BLOCK_SIZE_C, channels)
    end_h = tl.minimum(start_h + BLOCK_SIZE_H, out_height)
    end_w = tl.minimum(start_w + BLOCK_SIZE_W, out_width)
    
    # Process each position in the outputtile
    for c in range(start_c, end_c):
        for oh in range(start_h, end_h):
            for ow in range(start_w, end_w):
                # Map output coordinates to input coordinates
                input_h = oh / (out_height - 1) * (in_height - 1) if out_height > 1 else 0
                input_w = ow / (out_width - 1) * (in_width - 1) if out_width > 1 else 0
                
                # Get integer coordinates and weights for bilinear interpolation
                h0 = tl.floor(input_h)
                h1 = h0 + 1
                w0 = tl.floor(input_w) 
                w1 = w0 + 1
                
                # Calculate interpolation weights
                y_weight_upper = input_h - h0
                y_weight_lower = h1 - input_h
                x_weight_right = input_w - w0
                x_weight_left = w1 - input_w
                
                # Clamp coordinates to valid input range
                coord_h0 = tl.max(0, tl.min(h0, in_height - 1)).to(tl.int32)
                coord_h1 = tl.max(0, tl.min(h1, in_height - 1)).to(tl.int32)
                coord_w0 = tl.max(0, tl.min(w0, in_width - 1)).to(tl.int32)
                coord_w1 = tl.max(0, tl.min(w1, in_width - 1)).to(tl.int32)
                
                # Load the 4 nearest neighbor values from input
                I00 = tl.load(input_ptr + (0, c, coord_h0, coord_w0))
                I01 = tl.load(input_ptr + (0, c, coord_h0, coord_w1))
                I10 = tl.load(input_ptr + (0, c, coord_h1, coord_w0))
                I11 = tl.load(input_ptr + (0, c, coord_h1, coord_w1))
                
                # Perform bilinear interpolation
                interpolated = (
                    I00 * y_weight_lower * x_weight_left +
                    I01 * y_weight_lower * x_weight_right +
                    I10 * y_weight_upper * x_weight_left +
                    I11 * y_weight_upper * x_weight_right
                )
                
                # Load corresponding multiplier value
                mult_val = tl.load(multiply_ptr + (0, c, oh, ow))
                
                # Store interpolated and multiplied result
                result = interpolated * mult_val
                
                # Store result (Triton handles dtype conversion automatically)
                out_offset = out_ptr + (0, c, oh, ow)
                tl.store(out_offset, result)

@torch.fx.wrap  
def optimized_interpolate_multiply(input_tensor, multiply_tensor):
    """Wrapper for optimized bilinear interpolation + multiplication"""
    batch = input_tensor.shape[0]
    channels = input_tensor.shape[1] 
    in_height = input_tensor.shape[2]
    in_width = input_tensor.shape[3]
    out_height = multiply_tensor.shape[2]
    out_width = multiply_tensor.shape[3]
    
    out = torch.empty((batch, channels, out_height, out_width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimized block sizes based on tensor dimensions 
    BLOCK_SIZE_C = 32  # Process multiple channels together
    BLOCK_SIZE_H = 64  # Height tile size
    BLOCK_SIZE_W = 32  # Width tile size
    
    # Calculate number of programs needed
    num_programs_c = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_programs_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H 
    num_programs_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    optimized_interpolate_multiply_kernel[(num_programs_c, num_programs_h, num_programs_w)](
        input_ptr=input_tensor,
        multiply_ptr=multiply_tensor, 
        out_ptr=out,
        batch=batch,
        channels=channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out

def replacement_func():
    """Return the optimized interpolation + multiplication function"""
    return optimized_interpolate_multiply