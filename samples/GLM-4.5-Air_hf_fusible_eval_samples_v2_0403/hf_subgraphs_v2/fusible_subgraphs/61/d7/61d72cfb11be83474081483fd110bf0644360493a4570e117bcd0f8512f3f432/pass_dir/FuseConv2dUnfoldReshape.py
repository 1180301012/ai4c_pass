import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv_unfold_kernel(
    input_ptr,          # [1, 256, 32, 32] 
    weight_ptr,         # [128, 256, 1, 1]
    output_ptr,         # [1, 128, 4, 1024]
    input_batch,        
    input_channels,     
    input_height,       
    input_width,        
    output_channels,    
    unfold_h,           
    unfold_w,           
    # For conv2d calculation
    n_elems_per_output_channel,
    # Stride and dilation
    h_stride, w_stride,
    h_dilation, w_dilation,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2) 
    pid_w = tl.program_id(3)
    
    # Each program processes one output channel and one spatial position in the unfolded result
    oc = pid_m  # output channel
    unfold_pos = pid_n * (BLOCK_SIZE_N * unfold_h * unfold_w) + tl.arange(0, BLOCK_SIZE_N * unfold_h * unfold_w)
    
    # Map unfolded position to original input position
    unfold_hw_idx = unfold_pos // (unfold_h * unfold_w)
    unfold_local_idx = unfold_pos % (unfold_h * unfold_w)
    unfold_i = unfold_local_idx // unfold_w  
    unfold_j = unfold_local_idx % unfold_w
    
    input_i = pid_h * h_dilation * h_stride + unfold_i
    input_j = pid_w * w_dilation * w_stride + unfold_j
    
    # Ensure we're within bounds
    mask_i = input_i < input_height
    mask_j = input_j < input_width
    mask_out = mask_i & mask_j
    
    # Load weight for this output channel (1x1 kernel)
    weight_offset = oc * input_channels * 1 * 1
    weight_val = tl.load(weight_ptr + weight_offset, mask=True)
    
    # Load input patch around this position
    input_values = tl.empty((BLOCK_SIZE_N,), dtype=tl.float32)
    for idx in range(BLOCK_SIZE_N):
        input_idx = idx
        if input_idx < input_channels:
            global_offset = input_idx * input_height * input_width + input_i * input_width + input_j
            input_values[idx] = tl.load(input_ptr + global_offset, mask=mask_out)
        else:
            input_values[idx] = 0.0
    
    # Compute output: conv2d result for this patch
    # For 1x1 conv, it's just weighted sum over channels
    output_val = tl.sum(input_values * weight_val)
    
    # Store to unfolded output
    unfold_total_pos = pid_h * input_width * unfold_w + pid_w * unfold_w + unfold_pos
    global_output_offset = oc * (input_height // h_stride) * (input_width // w_stride) * unfold_h * unfold_w + unfold_total_pos
    
    if unfold_total_pos < (input_height // h_stride) * (input_width // w_stride) * unfold_h * unfold_w:
        tl.store(output_ptr + global_output_offset, output_val, mask=True)

@torch.fx.wrap
def fused_conv_unfold(input, weight):
    # Get input dimensions
    batch, channels, height, width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions after unfold operation
    # Withunfold (2, 2) and stride (2, 2) on conv2d output [1, 128, 32, 32]
    # Result is [1, 128, 4, 1024] after reshape
    
    # Grid setup
    grid_m = out_channels
    grid_n = (channels * height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_h = height // 2  # stride=2
    grid_w = width // 2    # stride=2
    
    # Calculate total out elements for the final reshaped tensor
    unfolded_h = 2  # kernel size
    unfolded_w = 2  # kernel size
    spatial_positions = (height // 2) * (width // 2)
    unfold_elements_per_pos = unfolded_h * unfolded_w
    
    # Create output tensor
    unfolded_size = out_channels * spatial_positions * unfold_elements_per_pos
    output = torch.empty(unfolded_size, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    fused_conv_unfold_kernel[(
        grid_m,
        (grid_n + 255) // 256,  # ceil division for grid size
        grid_h,
        grid_w
    )](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        input_batch=batch,
        input_channels=channels,
        input_height=height,
        input_width=width,
        output_channels=out_channels,
        unfold_h=unfolded_h,
        unfold_w=unfolded_w,
        n_elems_per_output_channel=unfolded_h * unfolded_w,
        h_stride=2, w_stride=2,
        h_dilation=1, w_dilation=1,
        BLOCK_SIZE_M=128,  # Process multiple output channels per program
        BLOCK_SIZE_N=256,  # Process multiple input elements per program
        BLOCK_SIZE_K=1,    # Doesn't matter for 1x1 conv
    )
    
    # Reshape to final output format
    final_output = output.reshape(1, out_channels, unfolded_h, -1)
    return final_output

def replacement_func():
    return fused_conv_unfold