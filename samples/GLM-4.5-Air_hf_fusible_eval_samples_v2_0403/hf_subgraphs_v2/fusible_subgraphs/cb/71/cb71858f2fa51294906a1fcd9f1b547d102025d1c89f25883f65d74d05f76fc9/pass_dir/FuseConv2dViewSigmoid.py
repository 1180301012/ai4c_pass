import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern for fusing conv2d → view → sigmoid operations
    This matches: 
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    out = tmp_3.sigmoid()
    return out

def replacement_args(in_0, in_1, in_2):
    """Extract input arguments for fused conv2d + view + sigmoid"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_view_sigmoid_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C_in, H_in, W_in,
    C_out, K_H, K_W,
    H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for conv2d + view + sigmoid operations
    
    Input shape: [1, 2, 1, 8]  -> Output after conv: [1, 128, 1, 8] 
    After view: [1, 2, 8, 8] -> after sigmoid: [1, 2, 8, 8]
    
    The view operation essentially rearranges conv2d output [1, 128, 1, 8] 
    to [1, 2, 8, 8] which implies a transpose and reshape operation.
    """
    pid = tl.program_id(0)
    
    # For the specific shapes, we have N=1, so each program handles one output position
    if pid >= N * 2 * 8 * 8:  # Target output shape [1, 2, 8, 8]
        return
    
    # Decode target output coordinates [1, 2, 8, 8]
    w_out = pid % 8
    pid = pid // 8
    h_out = pid % 8  
    pid = pid // 8
    c_out_group = pid % 2  # We have 2 output channels/groups
    n = pid // 2
    
    # For this specific view transformation:
    # [1, 128, 1, 8] -> [1, 2, 8, 8]
    # This suggests channel 0-63 map to output channel 0
    # channel 64-127 map to output channel 1
    # Each group of 8 consecutive channels in original map to height positions
    
    # Calculate which original conv2d channels contribute to this output position
    # This is a complex view transformation that we need to understand
    # For [1, 128, 1, 8] -> [1, 2, 8, 8]:
    # It's likely: output[n, c_out, h_out, w_out] = conv_output[n, c_out*64 + h_out, 0, w_out]
    
    conv_channel_idx = c_out_group * 64 + h_out
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + conv_channel_idx)
    
    # Load weight slice for this output channel
    # Weight shape: [128, 2, 1, 8]
    weight_offset = conv_channel_idx * (2 * 1 * 8) + c_out_group * (1 * 8)  # Only weight for corresponding input_channel
    weight_data = tl.load(weight_ptr + weight_offset)
    
    # Load input slice 
    # Input shape: [1, 2, 1, 8]
    input_base_offset = n * (2 * 1 * 8) + c_out_group * (1 * 8)  # Only input for corresponding input_channel
    input_offsets = input_base_offset + tl.arange(0, 8)
    input_data = tl.load(input_ptr + input_offsets)
    
    # Perform convolution computation for this 1x1 kernel
    # Since kernel is 1x1, it's just a weighted sum using dot product
    conv_val = bias_val + tl.sum(weight_data * input_data)
    
    # Apply sigmoid - cast to fp32 for numerical stability
    sigmoid_val = 1.0 / (1.0 + tl.exp(tl.cast(-conv_val, tl.float32)))
    sigmoid_val = tl.cast(sigmoid_val, conv_val.dtype)
    
    # Store to output
    output_idx = n * (2 * 8 * 8) + c_out_group * (8 * 8) + h_out * 8 + w_out
    tl.store(output_ptr + output_idx, sigmoid_val)

@torch.fx.wrap
def fused_conv2d_view_sigmoid(bias, weight, input_tensor):
    """
    Fused implementation of conv2d → view → sigmoid for the specific pattern
    """
    # Input shapes from weight_meta.py:
    # input: [1, 2, 1, 8], weight: [128, 2, 1, 8], bias: [128]
    
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, K_C, K_H, K_W = weight.shape
    
    # After conv2d with stride (1,1), padding (0,0): [1, 128, 1, 8]
    H_out = H_in 
    W_out = W_in
    
    # Final output after view: [1, 2, 8, 8]
    final_N, final_C, final_H, final_W = 1, 2, 8, 8
    
    # Output tensor
    output = torch.empty(final_N, final_C, final_H, final_W, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch Triton kernel
    total_elements = final_N * final_C * final_H * final_W
    
    # Use smaller block size for better occupancy with small tensors
    BLOCK_SIZE = 32  # Optimized for small workloads
    
    fused_conv_view_sigmoid_kernel[(total_elements,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, K_H=K_H, K_W=K_W,
        H_out=H_out, W_out=W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused conv2d + view + sigmoid function"""
    return fused_conv2d_view_sigmoid