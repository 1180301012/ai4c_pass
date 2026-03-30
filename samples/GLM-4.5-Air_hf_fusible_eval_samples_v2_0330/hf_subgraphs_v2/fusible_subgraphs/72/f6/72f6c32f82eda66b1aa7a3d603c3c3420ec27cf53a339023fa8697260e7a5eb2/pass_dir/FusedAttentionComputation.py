import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation pipeline
def pattern(in_0, in_1):
    # Conv2D operation
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    
    # Padding operation
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    
    # First unfold operation (height dimension)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    
    # Second unfold operation (width dimension)  
    tmp_4 = tmp_3.unfold(3, 12, 8)
    
    # Determine reshape parameters based on input dimensions
    # For bfloat16/float32: reshape(8, 48, 4, -1)
    # For float16: reshape(8, 80, 4, -1)
    # We need to determine which one to use based on the tensor shape
    if tmp_4.shape[1] == 640:  # float16 case
        tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    else:  # bfloat16/float32 case  
        tmp_5 = tmp_4.reshape(8, 48, 4, -1)
    
    # Permute dimensions
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    
    # Determine split parameters based on the last dimension size
    # For bfloat16/float32: split([16, 32])
    # For float16: split([16, 64])
    last_dim = tmp_6.shape[-1]
    if last_dim == 144:  # float16 case
        split = torch.functional.split(tmp_6, [16, 64], dim = -1)
    else:  # bfloat16/float32 case
        split = torch.functional.split(tmp_6, [16, 32], dim = -1)
    
    tmp_8 = split[0]
    tmp_9 = split[1]
    
    # Transpose for first output
    tmp_10 = tmp_8.transpose(-1, -2)
    
    return (tmp_10, tmp_9)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel using Triton
@triton.jit
def fused_attention_kernel(
    weight_ptr,           # [out_channels, in_channels, 1, 1] - weight tensor
    input_ptr,            # [batch, in_channels, height, width] - input tensor  
    output1_ptr,          # First output tensor 
    output2_ptr,          # Second output tensor
    batch,                # Batch size (should be 1)
    in_channels,          # Input channels (512)
    out_channels,         # Output channels (640 for float16, 384 for bfloat16/float32)
    input_h,              # Input height (16)
    input_w,              # Input width (16)
    reshape_dim2,         # Intermediate reshape dimension (80 or 48)
    split_sizes,          # Split sizes [16, 64] or [16, 32]
    last_dim,             # Last dimension before split (144 or 80)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output dimensions for generic case
    # After conv+pad: [1, out_channels, 20, 20]
    # After unfold+unfold: [1, out_channels, 2, 2, 12, 12]  
    # After reshape: [8, reshape_dim2, 4, last_dim]
    # After permute: [8, 4, last_dim, reshape_dim2]
    # After split: [8, 4, split_sizes[0], reshape_dim2] and [8, 4, split_sizes[1], reshape_dim2]
    
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
            
            # Generic handling for both data types
            elem_idx = h_idx * 12 * 12 + wy * 12 + wx
            
            # Determine which split this element goes to
            if elem_idx < split_sizes[0]:
                # First slice goes to output1
                output1_offset = (block_idx * 4 + h_idx) * split_sizes[0] * reshape_dim2 + elem_idx * reshape_dim2 + (channel_idx % 4)
                tl.store(output1_ptr + output1_offset, conv_val)
            elif elem_idx < last_dim:
                # Second slice goes to output2
                elem_idx_in_split2 = elem_idx - split_sizes[0] 
                output2_offset = (block_idx * 4 + h_idx) * split_sizes[1] * reshape_dim2 + elem_idx_in_split2 * reshape_dim2 + (channel_idx % 4)
                tl.store(output2_ptr + output2_offset, conv_val)

@torch.fx.wrap
def fused_attention_computation(weight, input):
    # Get input shapes
    batch, in_channels, input_h, input_w = input.shape
    out_channels, _, _, _ = weight.shape
    
    # Determine which data type we're dealing with and set output shapes accordingly
    if out_channels == 640:  # float16 case
        output1_shape = [8, 4, 16, 80]   # split [16, 64]
        output2_shape = [8, 4, 64, 80]
    else:  # bfloat16/float32 case
        output1_shape = [8, 4, 16, 48]   # split [16, 32]  
        output2_shape = [8, 4, 32, 48]
    
    # Create output tensors
    output1 = torch.empty(output1_shape, dtype=input.dtype, device=input.device)
    output2 = torch.empty(output2_shape, dtype=input.dtype, device=input.device)
    
    # Launch kernel with data-specific parameters
    BLOCK_SIZE = 1024
    
    # Add configuration for different data types
    if out_channels == 640:  # float16
        reshape_dim2 = 80
        split_sizes = [16, 64]
        last_dim = 144
    else:  # bfloat16/float32
        reshape_dim2 = 48
        split_sizes = [16, 32] 
        last_dim = 80
    
    # Calculate total number of programs
    total_elements = 8 * out_channels * 2 * 2  # 8 blocks * channels * 2*2 spatial positions
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_kernel[(num_programs,)](
        weight_ptr=weight,
        input_ptr=input,
        output1_ptr=output1,
        output2_ptr=output2,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        input_h=input_h,
        input_w=input_w,
        reshape_dim2=reshape_dim2,
        split_sizes=split_sizes,
        last_dim=last_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Transpose final output - the exact transpose depends on the shape
    if out_channels == 640:  # float16
        output1 = output1.transpose(-1, -2)  # [8, 4, 16, 80] -> [8, 4, 80, 16]
        output1 = output1.transpose(1, 2)   # [8, 4, 80, 16] -> [8, 80, 4, 16]
        output1 = output1.transpose(-2, -1) # [8, 80, 4, 16] -> [8, 80, 16, 4]
    else:  # bfloat16/float32
        output1 = output1.transpose(-1, -2)  # [8, 4, 16, 48] -> [8, 4, 48, 16]
        output1 = output1.transpose(1, 2)   # [8, 4, 48, 16] -> [8, 48, 4, 16]
        output1 = output1.transpose(-2, -1) # [8, 48, 4, 16] -> [8, 48, 16, 4]
    
    return output1, output2

# Replacement function
def replacement_func():
    return fused_attention_computation