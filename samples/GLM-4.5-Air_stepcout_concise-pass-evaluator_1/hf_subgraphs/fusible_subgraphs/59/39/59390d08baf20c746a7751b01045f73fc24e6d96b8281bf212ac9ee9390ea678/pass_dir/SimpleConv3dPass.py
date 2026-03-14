import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    """Pattern matching the conv3d operation"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_3 = torch.conv3d(in_3, tmp_1, tmp_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    return tmp_3

def replacement_args(in_0, in_1, in_3):
    """Extract arguments for the replacement kernel"""
    return (in_0, in_1, in_3)

@triton.jit
def simple_conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    input_depth, input_height, input_width,
    output_depth, output_height, output_width,
):
    """Simple conv3d kernel"""
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    
    # Each program handles one output position
    if pid >= batch_size * out_channels * output_depth * output_height * output_width:
        return
        
    # Calculate output coordinates
    batch_idx = pid // (out_channels * output_depth * output_height * output_width)
    remainder = pid % (out_channels * output_depth * output_height * output_width)
    
    oc = remainder // (output_depth * output_height * output_width)
    remainder = remainder % (output_depth * output_height * output_width)
    
    od = remainder // (output_height * output_width)
    oh = remainder // output_width
    ow = remainder % output_width
    
    # Calculate corresponding input coordinates
    id_base = od * 2  # stride = 2 for depth
    ih_base = oh * 16  # stride = 16 for height  
    iw_base = ow * 16  # stride = 16 for width
    
    # Accumulate convolution result
    result = 0.0
    
    for ic in range(in_channels):
        for kd in range(2):  # kernel depth
            for kh in range(16):  # kernel height
                for kw in range(16):  # kernel width
                    id_pos = id_base + kd
                    ih_pos = ih_base + kh
                    iw_pos = iw_base + kw
                    
                    if (id_pos < input_depth):
                        if (ih_pos < input_height):
                            if (iw_pos < input_width):
                                # Calculate input offset: [B, C, D, H, W]
                                input_offset = (batch_idx * in_channels * input_depth * input_height * input_width +
                                              ic * input_depth * input_height * input_width +
                                              id_pos * input_height * input_width +
                                              ih_pos * input_width +
                                              iw_pos)
                                
                                # Calculate weight offset: [C_out, C_in, K_d, K_h, K_w] 
                                weight_offset = (oc * in_channels * 2 * 16 * 16 +
                                               ic * 2 * 16 * 16 +
                                               kd * 16 * 16 +
                                               kh * 16 +
                                               kw)
                                
                                input_val = tl.load(input_ptr + input_offset)
                                weight_val = tl.load(weight_ptr + weight_offset)
                                result += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + oc)
    result += bias_val
    
    # Store result
    output_offset = (batch_idx * out_channels * output_depth * output_height * output_width +
                    oc * output_depth * output_height * output_width +
                    od * output_height * output_width +
                    oh * output_width +
                    ow)
    tl.store(output_ptr + output_offset, result)

@torch.fx.wrap  
def simple_conv3d(in_0, in_1, in_3):
    """Simple conv3d replacement"""
    batch_size, in_channels, input_depth, input_height, input_width = in_3.shape
    in_channels_, out_channels, _, _, _ = in_1.shape
    
    output_depth = input_depth // 2
    output_height = input_height // 16
    output_width = input_width // 16
    
    output_shape = (batch_size, out_channels, output_depth, output_height, output_width)
    output = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    total_elements = batch_size * out_channels * output_depth * output_height * output_width
    simple_conv3d_kernel[(total_elements, 1, 1)](in_3, in_1, in_0, output, batch_size, in_channels, out_channels,
                                                input_depth, input_height, input_width,
                                                output_depth, output_height, output_width)
    
    return output

def replacement_func():
    """Return the replacement function"""
    return simple_conv3d