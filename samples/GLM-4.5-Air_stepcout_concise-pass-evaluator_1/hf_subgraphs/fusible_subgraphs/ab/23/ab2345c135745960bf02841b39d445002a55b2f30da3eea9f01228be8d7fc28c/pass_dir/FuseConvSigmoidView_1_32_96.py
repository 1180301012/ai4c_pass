import torch
import triton
import triton.language as tl

# Pattern: Conv2D + Sigmoid + View fusion
def pattern(in_3, in_1, in_0):
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    return tmp_4

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Optimized kernel: Fused grouped conv + sigmoid + reshape
@triton.jit
def fused_conv_sigmoid_view_kernel(
    input_ptr,  # [1, 32, 1, 1] - flattened to [32]
    weight_ptr, # [96, 8, 1, 1] - flattened to [768]
    bias_ptr,   # [96]
    output_ptr, # [96]
    groups: tl.constexpr,
):
    # Each program handles one output channel
    pid = tl.program_id(0)
    
    # Only process if within valid range
    if pid < 96:
        # Load bias for this output channel
        out_val = tl.load(bias_ptr + pid)
        
        # Determine group for this output channel
        group_id = pid // (96 // groups)
        
        # Calculate input channels for this group
        input_ch_per_group = 32 // groups
        start_input_ch = group_id * input_ch_per_group
        
        # Compute grouped convolution: sum over input channels in the group
        for j in range(input_ch_per_group):
            # Load input channel
            input_ch = start_input_ch + j
            input_val = tl.load(input_ptr + input_ch)
            
            # Load corresponding weight
            weight_offset = pid * input_ch_per_group + j
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Accumulate: weight * input
            out_val += weight_val * input_val
        
        # Apply sigmoid activation
        sigmoid_val = 1.0 / (1.0 + tl.exp(-out_val))
        
        # Store result
        tl.store(output_ptr + pid, sigmoid_val)

@torch.fx.wrap
def fused_conv_sigmoid_view(in_3, in_1, in_0):
    # Flatten tensors for efficient memory access
    input_flat = in_3.view(-1)  # [1, 32, 1, 1] -> [32]
    weight_flat = in_1.view(-1) # [96, 8, 1, 1] -> [768]
    
    # Create output tensor [96]
    output = torch.empty(96, dtype=torch.float32, device=in_3.device)
    
    # Launch one program per output channel (96 total)
    fused_conv_sigmoid_view_kernel[(96,)](
        input_ptr=input_flat,
        weight_ptr=weight_flat,
        bias_ptr=in_0,
        output_ptr=output,
        groups=4,
    )
    
    # Reshape to view format [1, -1, 1, 1]
    return output.view(1, -1, 1, 1)

def replacement_func():
    return fused_conv_sigmoid_view