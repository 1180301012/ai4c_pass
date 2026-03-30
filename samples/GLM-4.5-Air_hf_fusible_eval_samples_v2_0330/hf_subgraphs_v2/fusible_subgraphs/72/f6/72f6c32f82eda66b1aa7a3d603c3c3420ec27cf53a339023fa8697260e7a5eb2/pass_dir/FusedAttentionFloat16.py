import torch
import triton
import triton.language as tl

# Pattern matching function for float16 variant
def pattern(in_0, in_1):
    # Conv2D operation
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    
    # Padding operation
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    
    # First unfold operation (height dimension)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    
    # Second unfold operation (width dimension)  
    tmp_4 = tmp_3.unfold(3, 12, 8)
    
    # Reshape operation for float16
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    
    # Permute dimensions
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    
    # Split into two outputs for float16
    split = torch.functional.split(tmp_6, [16, 64], dim = -1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    
    # Transpose for first output
    tmp_10 = tmp_8.transpose(-1, -2)
    
    return (tmp_10, tmp_9)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for float16 using Triton
@triton.jit
def fused_attention_float16_kernel(
    weight_ptr,           # [640, 512, 1, 1] - weight tensor
    input_ptr,            # [1, 512, 16, 16] - input tensor  
    output1_ptr,          # First output tensor [8, 4, 16, 80] 
    output2_ptr,          # Second output tensor [8, 4, 64, 80]
    batch,                # Batch size (should be 1)
    in_channels,          # Input channels (512)
    out_channels,         # Output channels (640)
    input_h,              # Input height (16)
    input_w,              # Input width (16)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output dimensions for float16
    # After conv+pad: [1, 640, 20, 20]
    # After unfold+unfold: [1, 640, 2, 2, 12, 12]  
    # After reshape: [8, 80, 4, 144]
    # After permute: [8, 4, 144, 80]
    # After split: [8, 4, 16, 80] and [8, 4, 64, 80]
    
    block_idx = pid // (out_channels * 2 * 2)
    channel_idx = (pid // (2 * 2)) % out_channels
    h_idx = (pid // 2) % 2
    w_idx = pid % 2
    
    if block_idx >= 8 or channel_idx >= out_channels or h_idx >= 2 or w_idx >= 2:
        return
        
    # Calculate input position
    base_input_offset = block_idx * out_channels * input_h * input_w + channel_idx * input_h * input_w + h_idx * input_w + w_idx
    
    # Process 12x12 sliding window
    for wy in range(12):
        for wx in range(12):
            # Calculate conv output for this position using the 1x1 kernel
            conv_val = 0.0
            for ic in range(in_channels):
                input_offset = base_input_offset + ic * input_h * input_w
                input_val = tl.load(input_ptr + input_offset)
                
                # Load weight value for this input channel and output channel
                weight_offset = channel_idx * in_channels + ic
                weight_val = tl.load(weight_ptr + weight_offset)
                conv_val += input_val * weight_val
            
            # This conv_val represents the [h_idx, w_idx, wy, wx] position in the unfolded tensor
            
            # Element index in the unfolded window
            elem_idx = h_idx * 12 * 12 + wy * 12 + wx
            
            # Split handling for float16: [16, 64]
            if elem_idx < 16:
                # First 16 elements go to output1
                output1_offset = (block_idx * 4 + h_idx) * 16 * 80 + elem_idx * 80 + (channel_idx % 4)
                tl.store(output1_ptr + output1_offset, conv_val)
            elif elem_idx < 144:
                # Elements 16-143 go to output2 (128 elements, keeping first 64)
                elem_idx_in_split2 = elem_idx - 16 
                output2_offset = (block_idx * 4 + h_idx) * 64 * 80 + min(elem_idx_in_split2, 63) * 80 + (channel_idx % 4)
                tl.store(output2_ptr + output2_offset, conv_val)

@torch.fx.wrap
def fused_attention_float16(weight, input):
    # Get input shapes
    batch, in_channels, input_h, input_w = input.shape
    out_channels, _, _, _ = weight.shape
    
    # Output shapes for float16
    output1_shape = [8, 4, 16, 80]
    output2_shape = [8, 4, 64, 80]
    
    # Create output tensors
    output1 = torch.empty(output1_shape, dtype=input.dtype, device=input.device)
    output2 = torch.empty(output2_shape, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    
    # Calculate total number of programs
    total_elements = 8 * out_channels * 2 * 2  # 8 blocks * channels * 2*2 spatial positions
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_float16_kernel[(num_programs,)](
        weight_ptr=weight,
        input_ptr=input,
        output1_ptr=output1,
        output2_ptr=output2,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        input_h=input_h,
        input_w=input_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Transpose final output for float16
    output1 = output1.transpose(-1, -2)  # [8, 4, 16, 80] -> [8, 4, 80, 16]
    output1 = output1.transpose(1, 2)   # [8, 4, 80, 16] -> [8, 80, 4, 16]
    output1 = output1.transpose(-2, -1) # [8, 80, 4, 16] -> [8, 80, 16, 4]
    
    return output1, output2

# Replacement function
def replacement_func():
    return fused_attention_float16