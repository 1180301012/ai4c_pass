import torch
import triton
import triton.language as tl


def pattern(pool_input, concat_input):
    # Match the exact computation pattern from model.py
    # max_pool2d -> interpolate -> cat
    # This handles different input tensors and target sizes
    tmp_4 = torch.nn.functional.max_pool2d(pool_input, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (64, 64), None, 'bilinear', False)
    tmp_6 = torch.cat([concat_input, tmp_5], 1)
    return tmp_6


def replacement_args(pool_input, concat_input):
    return (pool_input, concat_input)


@triton.jit
def optimized_pool_interp_cat_kernel(
    input_ptr,
    concat_tensor_ptr,
    output_ptr,
    input_N, input_C, input_H, input_W,
    concat_N, concat_C, concat_H, concat_W,
    output_N, output_C, output_H, output_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (output_N * output_C * output_H * output_W)
    
    # Initialize output with concat tensor values where appropriate
    if offsets < (concat_N * concat_C * concat_H * concat_W):
        # Copy from concat tensor for the first concat_C channels
        concat_offset = offsets
        concat_mask = concat_offset < (concat_N * concat_C * concat_H * concat_W)
        concat_val = tl.load(concat_tensor_ptr + concat_offset, mask=concat_mask, other=0.0)
        tl.store(output_ptr + offsets, concat_val, mask=concat_mask & mask)
    
    # Process pool + interp for the remaining channels
    interp_start_offset = concat_N * concat_C * concat_H * concat_W
    if offsets < (output_N * output_C * output_H * output_W):
        if offsets >= interp_start_offset:
            # Calculate interpolation position
            remaining_offset = offsets - interp_start_offset
            interp_N = remaining_offset // (output_C * output_H * output_W)
            interp_C = (remaining_offset // (output_H * output_W)) % output_C
            interp_H = (remaining_offset // output_W) % output_H
            interp_W = remaining_offset % output_W
            
            # Map back to input coordinates (upscaling by 2)
            src_H = interp_H * 2
            src_W = interp_W * 2
            src_mask = (src_H < input_H) & (src_W < input_W)
            
            # Bilinear interpolation weights
            y_f = interp_H * 2
            x_f = interp_W * 2
            y_low = tl.math.floor(y_f)
            y_high = y_low + 1
            x_low = tl.math.floor(x_f)
            x_high = x_low + 1
            
            # Ensure coordinates are within bounds
            y_low = tl.minimum(y_low, input_H - 1)
            y_high = tl.minimum(y_high, input_H - 1)
            x_low = tl.minimum(x_low, input_W - 1)
            x_high = tl.minimum(x_high, input_W - 1)
            
            # Calculate weights
            y_weight = y_f - y_low
            x_weight = x_f - x_low
            w_00 = (1 - y_weight) * (1 - x_weight)
            w_01 = (1 - y_weight) * x_weight
            w_10 = y_weight * (1 - x_weight)
            w_11 = y_weight * x_weight
            
            # Load four neighboring pixels
            src_offsets = [
                interp_N * input_C * input_H * input_W + interp_C * input_H * input_W + y_low * input_W + x_low,
                interp_N * input_C * input_H * input_W + interp_C * input_H * input_W + y_low * input_W + x_high,
                interp_N * input_C * input_H * input_W + interp_C * input_H * input_W + y_high * input_W + x_low,
                interp_N * input_C * input_H * input_W + interp_C * input_H * input_W + y_high * input_W + x_high,
            ]
            
            val_00 = tl.load(input_ptr + src_offsets[0], mask=src_mask, other=0.0)
            val_01 = tl.load(input_ptr + src_offsets[1], mask=src_mask, other=0.0)
            val_10 = tl.load(input_ptr + src_offsets[2], mask=src_mask, other=0.0)
            val_11 = tl.load(input_ptr + src_offsets[3], mask=src_mask, other=0.0)
            
            # Bilinear interpolation
            interp_val = (w_00 * val_00 + w_01 * val_01 + w_10 * val_10 + w_11 * val_11)
            tl.store(output_ptr + offsets, interp_val, mask=mask)


@torch.fx.wrap
def optimized_pool_interp_cat(pool_input, concat_input):
    # Get tensor properties
    input_shape = pool_input.shape  # [N, C, H, W] -> after maxpool: [N, C, H//2, W//2]
    concat_shape = concat_input.shape  # [N, C2, H2, W2]
    
    # Output shape after interpolation: [N, C, H2, W2] where H2=W2=64 (from the pattern)
    output_shape = [input_shape[0], input_shape[1], 64, 64]
    
    # Calculate total elements
    output_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    total_elements = concat_shape[0] * concat_shape[1] * concat_shape[2] * concat_shape[3] + output_elements
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(concat_shape[0], concat_shape[1] + input_shape[1], concat_shape[2], concat_shape[3], dtype=pool_input.dtype, device=pool_input.device)
    
    # Launch kernel
    optimized_pool_interp_cat_kernel[(num_programs,)](
        input_ptr=pool_input,
        concat_tensor_ptr=concat_input,
        output_ptr=output,
        input_N=input_shape[0], input_C=input_shape[1], input_H=input_shape[2], input_W=input_shape[3],
        concat_N=concat_shape[0], concat_C=concat_shape[1], concat_H=concat_shape[2], concat_W=concat_shape[3],
        output_N=output_shape[0], output_C=output_shape[1], output_H=output_shape[2], output_W=output_shape[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_pool_interp_cat