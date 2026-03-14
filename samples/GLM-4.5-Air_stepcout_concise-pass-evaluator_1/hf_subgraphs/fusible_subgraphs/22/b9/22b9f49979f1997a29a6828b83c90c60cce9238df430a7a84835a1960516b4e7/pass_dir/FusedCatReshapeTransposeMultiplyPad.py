import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching: cat + reshape + transpose + multiply + pad"""
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, -1, 576)  # Use -1 for flexible dim
    tmp_0 = None
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_1 = None
    tmp_3 = in_3 * tmp_2
    tmp_2 = None
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_3 = None
    return (tmp_4,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr,
    n0_batch, n0_channels, n0_height, n0_width,
    n1_batch, n1_channels, n1_height, n1_width, 
    n2_batch, n2_channels, n2_height, n2_width,
    n3_batch, n3_channels, n3_height, n3_width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel: concatenate, reshape, transpose, multiply, and pad"""
    
    # Calculate output dimensions based on the computation pattern
    concat_channels = n0_channels + n1_channels + n2_channels
    output_H = n0_height  # From original width after reshape
    output_W = 576  # Fixed from reshape pattern
    
    # Each program handles one output channel in the 8-channel dimension
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    if pid_m >= 8 or pid_h >= output_H + 1 or pid_w >= output_W:
        return
    
    # Shift to include the padded row
    h_offset = 1 if pid_h == output_H else 0
    h_idx = pid_h - h_offset
    
    # Load from in3 (the multiplier tensor) - shape [1, 8, W, H] after transpose
    # in3 has shape [1, 8, 576, 40] for first graph, etc.
    if pid_m < n3_channels and pid_w < n3_width and pid_h < n3_height:
        in3_val = tl.load(in3_ptr + pid_m * n3_height * n3_width + h_idx * n3_width + pid_w)
    else:
        in3_val = 0.0
    
    # Calculate which input tensor and position to load from for the concat operation
    # Concat happens along channel dimension, then reshape and transpose
    total_elements = n0_height * n0_width + n1_height * n1_width + n2_height * n2_width
    
    # For each position in the output (after transpose), find corresponding input
    linear_pos = h_idx * output_W + pid_w
    
    # Determine which input tensor this position comes from
    current_pos = 0
    in_val = 0.0
    
    if linear_pos < n0_height * n0_width:
        # From in_0
        local_h = linear_pos // n0_width
        local_w = linear_pos % n0_width
        in_val = tl.load(in0_ptr + local_h * n0_width + local_w)
    elif linear_pos - n0_height * n0_width < n1_height * n1_width:
        # From in_1  
        local_pos = linear_pos - n0_height * n0_width
        local_h = local_pos // n1_width
        local_w = local_pos % n1_width
        in_val = tl.load(in1_ptr + local_h * n1_width + local_w)
    else:
        # From in_2
        local_pos = linear_pos - n0_height * n0_width - n1_height * n1_width
        local_h = local_pos // n2_width
        local_w = local_pos % n2_width
        in_val = tl.load(in2_ptr + local_h * n2_width + local_w)
    
    # Apply multiplication and store result
    if pid_m < 8 and pid_h < output_H + 1 and pid_w < output_W:
        result = in_val * in3_val
        tl.store(out_ptr + pid_m * (output_H + 1) * output_W + pid_h * output_W + pid_w, result)

@torch.fx.wrap
def fused_computation(in_0, in_1, in_2, in_3):
    """Fused computation wrapper"""
    
    # Get input shapes
    shape_0 = in_0.shape
    shape_1 = in_1.shape  
    shape_2 = in_2.shape
    shape_3 = in_3.shape
    
    # Calculate output shape
    concat_channels = shape_0[1] + shape_1[1] + shape_2[1]
    output_H = shape_0[2]  # Original height becomes width after transpose
    output_W = min(576, shape_0[3] if len(shape_0) > 3 else 576)  # From reshape pattern
    final_height = output_H + 1  # From padding
    
    # Create output tensor
    output = torch.empty((1, 8, final_height, output_W), dtype=in_0.dtype, device=in_0.device)
    
    # Launch Triton kernel
    grid = lambda meta: (8, final_height, output_W)
    
    fused_kernel[grid](
        in0_ptr=in_0.data_ptr(),
        in1_ptr=in_1.data_ptr(),
        in2_ptr=in_2.data_ptr(),
        in3_ptr=in_3.data_ptr(),
        out_ptr=output.data_ptr(),
        n0_batch=shape_0[0], n0_channels=shape_0[1], n0_height=shape_0[2], n0_width=shape_0[3],
        n1_batch=shape_1[0], n1_channels=shape_1[1], n1_height=shape_1[2], n1_width=shape_1[3],
        n2_batch=shape_2[0], n2_channels=shape_2[1], n2_height=shape_2[2], n2_width=shape_2[3],
        n3_batch=shape_3[0], n3_channels=shape_3[1], n3_height=shape_3[2], n3_width=shape_3[3],
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=256
    )
    
    return output

def replacement_func():
    return fused_computation