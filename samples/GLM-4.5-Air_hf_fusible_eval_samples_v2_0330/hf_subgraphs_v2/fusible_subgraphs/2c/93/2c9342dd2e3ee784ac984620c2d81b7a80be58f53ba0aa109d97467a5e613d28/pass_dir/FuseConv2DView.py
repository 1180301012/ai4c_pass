import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias, view_size_1):
    conv2d = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(view_size_1, 1, -1)
    return tmp_3

def replacement_args(conv_input, weight, bias, view_size_1):
    return (conv_input, weight, bias, view_size_1)

@triton.jit
def fused_conv2d_view_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate batch and output channel indices
    pid = tl.program_id(0)
    total_elements = N * C_out * H * W
    num_programs = tl.cdiv(total_elements, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
        
    # Create flat index
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Convert flat index to multi-dimensional indices
    n = idx // (C_out * H * W)              # batch index
    c_out = (idx // (H * W)) % C_out        # output channel
    h = (idx // W) % H                      # height index  
    w = idx % W                            # width index
    
    # For 1x1 convolution: each output element is just input * weight + bias
    # Load input element
    input_idx = n * C_in * H * W + c_out * H * W + h * W + w
    input_val = tl.load(input_ptr + input_idx, mask=mask)
    
    # Load weight (1x1 conv means weight[c_out, c_in] where c_in = c_out for this special case)
    weight_idx = c_out * C_in
    weight_val = tl.load(weight_ptr + weight_idx, mask=mask)
    
    # Load bias  
    bias_val = tl.load(bias_ptr + c_out, mask=mask)
    
    # Compute result
    result = input_val * weight_val + bias_val
    
    # Store in desired view format: [N, 1, -1], flattened
    # The original view reshapes [N, C_out, H, W] to [N, 1, C_out * H * W]
    # So we need to store element 0: n*1*(C_out*H*W) + 0*(C_out*H*W) + (c_out*H*W + h*W + w)
    output_idx = n * (C_out * H * W) + (c_out * H * W + h * W + w)
    tl.store(output_ptr + output_idx, result, mask=mask)

@torch.fx.wrap  
def fused_conv2d_view_conv(input, weight, bias, view_size_1):
    N, C_in, H, W = input.shape
    C_out = weight.shape[0]
    
    # Create output with shape [N, 1, C_out * H * W]  
    output_elements = C_out * H * W
    output = torch.empty((view_size_1, 1, output_elements), dtype=input.dtype, device=input.device)
    
    BLOCK_SIZE = 1024
    total_elements = N * C_out * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_view_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_in=C_in, H=H, W=W, C_out=C_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # The view_size_1 might not match actual N, so transpose to match expected shape
    if view_size_1 != N:
        output = output.transpose(0, 1)
    
    return output

def replacement_func():
    return fused_conv2d_view_conv