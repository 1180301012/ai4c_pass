import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def fused_conv2d_sigmoid_kernel(
    output_ptr,     # output: [1, 2, 8, 8]
    input_ptr,      # in_2: [1, 2, 1, 8]
    weight_ptr,     # in_1: [128, 2, 1, 8] 
    bias_ptr,       # in_0: [128]
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
):
    # Each program handles one spatial location in the final output
    pid = tl.program_id(0)
    
    h_out = pid // W_out
    w_out = pid % W_out
    
    if h_out >= H_out or w_out >= W_out:
        return
    
    # For each output channel (only 2 in this case), compute the operation + sigmoid
    for c_out in range(2):
        # Initialize result for this output channel  
        result = 0.0
        
        # For each input channel (2 in this case)
        for c_in in range(2):  # C_in = 2
            # Load input value: [1, 2, 1, 8] -> [c_in, 0, w_out] position
            input_offset = (c_in * W_in) + w_out
            input_val = tl.load(input_ptr + input_offset)
            
            # Load weight: [128, 2, 1, 8] 
            # For output channel c_out, input channel c_in, at spatial w_out
            # We use bias_idx = c_out (since we only have 2 bias weights corresponding to 2 output channels)
            weight_offset = (c_out * 2 + c_in) * W_in + w_out  # 2 = C_in
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Compute dot product element
            result += weight_val * input_val
        
        # Load bias for this output channel  
        bias_val = tl.load(bias_ptr + c_out)
        
        # Add bias to result
        result += bias_val
        
        # Apply sigmoid
        sigmoid_result = 1.0 / (1.0 + tl.exp(-result))
        
        # Store result at [h_out, w_out, c_out] position in flattened [1, 2, 8, 8] tensor
        output_offset = (h_out * W_out * 2) + (w_out * 2) + c_out
        tl.store(output_ptr + output_offset, sigmoid_result)

@torch.fx.wrap
def fused_conv2d_sigmoid(in_2, in_1, in_0):
    # Input shapes: in_2[1, 2, 1, 8], in_1[128, 2, 1, 8], in_0[128]
    # Output shape: [1, 2, 8, 8]
    
    N, C_in, H_in, W_in = in_2.shape
    C_out, _, _, _ = in_1.shape
    
    # For our specific case, we know the target output shape
    output_shape = (1, 2, 8, 8)
    H_out, W_out = 8, 8  # Output spatial dimensions
    
    # Grid size: each program handles one spatial location in 8x8 output
    grid_size = H_out * W_out
    
    # Allocate output tensor
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    fused_conv2d_sigmoid_kernel[(grid_size,)](
        output,
        in_2,
        in_1,
        in_0,
        W_in,           # Input width (8)
        H_out,          # Output height (8)
        W_out,          # Output width (8)
    )
    
    return output

def replacement_func():
    return fused_conv2d_sigmoid