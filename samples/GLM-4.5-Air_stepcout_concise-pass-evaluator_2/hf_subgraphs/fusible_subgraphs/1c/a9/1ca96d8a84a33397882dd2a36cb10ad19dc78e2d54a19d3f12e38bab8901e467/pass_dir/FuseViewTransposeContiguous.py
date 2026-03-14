import torch
import triton
import triton.language as tl

def pattern(input_5, orig_shape_1, input_6, orig_shape_2):
    # Match the view -> transpose -> contiguous -> view pattern
    tmp_7 = input_5.view(orig_shape_1)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(orig_shape_1[0], orig_shape_1[1]*orig_shape_1[2], orig_shape_1[3], orig_shape_1[4])
    
    tmp_11 = input_6.view(orig_shape_2)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(orig_shape_2[0], orig_shape_2[1]*orig_shape_2[2], orig_shape_2[3], orig_shape_2[4])
    
    return tmp_8, tmp_10, tmp_12, tmp_14

def replacement_args(input_5, orig_shape_1, input_6, orig_shape_2):
    return (input_5, orig_shape_1, input_6, orig_shape_2)

@triton.jit
def fused_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size, feature_blocks, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output shape: [batch_size, feature_blocks * feature_size, height, width]
    output_features = feature_blocks * (1 if feature_blocks > 1 else 1)
    
    # Grid over output tensor
    x_offset = pid % width
    y_offset = (pid // width) % height
    f_offset = (pid // (width * height)) % output_features
    b_offset = pid // (width * height * output_features)
    
    mask = (b_offset < batch_size) & (f_offset < output_features) & (y_offset < height) & (x_offset < width)
    
    if mask:
        # Calculate input position in reshaped format
        input_features = feature_blocks
        input_f = f_offset if f_offset < input_features else 0
        input_b = b_offset
        input_x = x_offset
        input_y = y_offset
        
        # Load input value
        input_pos = ((input_b * input_features + input_f) * height + input_y) * width + input_x
        input_val = tl.load(input_ptr + input_pos)
        
        # Store output value (no transformation needed, just reshaping)
        output_pos = ((b_offset * output_features + f_offset) * height + y_offset) * width + x_offset
        tl.store(output_ptr + output_pos, input_val)

@torch.fx.wrap
def fused_reshape_view_contiguous(input_5, orig_shape_1, input_6, orig_shape_2):
    # Process first tensor: view(B, 2, 20, H, W) -> view(B, 40, H, W)
    batch_size_1 = orig_shape_1[0]
    input_features_1 = orig_shape_1[1]
    feature_size_1 = orig_shape_1[2] 
    height_1 = orig_shape_1[3]
    width_1 = orig_shape_1[4]
    output_features_1 = input_features_1 * feature_size_1
    
    # Output shape for first tensor
    out_shape_1 = (batch_size_1, output_features_1, height_1, width_1)
    out_10 = torch.empty(out_shape_1, dtype=input_5.dtype, device=input_5.device)
    
    if input_5.numel() > 0:
        BLOCK_SIZE = 1024
        total_output_elements = out_shape_1[0] * out_shape_1[1] * out_shape_1[2] * out_shape_1[3]
        num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        fused_reshape_kernel[(num_programs,)](
            input_ptr=input_5,
            output_ptr=out_10,
            batch_size=batch_size_1,
            feature_blocks=input_features_1,
            height=height_1, 
            width=width_1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Output shape for first intermediate (transpose result)
    out_shape_8 = (batch_size_1, feature_size_1, input_features_1, height_1, width_1)
    out_8 = input_5.reshape(out_shape_8)
    
    # Process second tensor: view(B, 2, 40, H, W) -> view(B, 80, H, W)  
    batch_size_2 = orig_shape_2[0]
    input_features_2 = orig_shape_2[1]
    feature_size_2 = orig_shape_2[2]
    height_2 = orig_shape_2[3]
    width_2 = orig_shape_2[4]
    output_features_2 = input_features_2 * feature_size_2
    
    # Output shape for second tensor
    out_shape_14 = (batch_size_2, output_features_2, height_2, width_2)
    out_14 = torch.empty(out_shape_14, dtype=input_6.dtype, device=input_6.device)
    
    if input_6.numel() > 0:
        total_output_elements = out_shape_14[0] * out_shape_14[1] * out_shape_14[2] * out_shape_14[3]
        num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        fused_reshape_kernel[(num_programs,)](
            input_ptr=input_6,
            output_ptr=out_14,
            batch_size=batch_size_2,
            feature_blocks=input_features_2,
            height=height_2,
            width=width_2,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Output shape for second intermediate (transpose result)
    out_shape_12 = (batch_size_2, feature_size_2, input_features_2, height_2, width_2)
    out_12 = input_6.reshape(out_shape_12)
    
    return out_8, out_10, out_12, out_14

def replacement_func():
    return fused_reshape_view_contiguous