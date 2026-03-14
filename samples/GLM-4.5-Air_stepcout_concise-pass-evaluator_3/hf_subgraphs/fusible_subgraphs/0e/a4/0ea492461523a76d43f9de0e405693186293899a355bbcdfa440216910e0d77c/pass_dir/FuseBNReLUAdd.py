import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Pattern matching: in_5 += in_4 -> batch_norm -> relu"""
    in_5 += in_4
    tmp_4 = in_5
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return tmp_4, tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def fused_bn_relu_add_kernel(
    input_ptr,  # input tensor in_5
    add_ptr,  # tensor to add (in_4)
    output_added_ptr,  # where to store result after addition (tmp_4)
    output_relu_ptr,  # where to store result after ReLU (tmp_6)
    running_mean_ptr,  # in_0
    running_var_ptr,  # in_1  
    weight_ptr,  # in_3
    bias_ptr,  # in_2
    n_elements,  # total elements in tensor
    H,  # height dimension
    W,  # width dimension
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: add + batch_norm + relu"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor in_5 and add tensor in_4
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    add_val = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition (tmp_4 = in_5 + in_4)
    x_added = x + add_val
    tl.store(output_added_ptr + offsets, x_added, mask=mask)
    
    # Calculate channel index for batch normalization
    # For shape [B, C, H, W], channel index = (offset // (H * W)) % C
    channel_idx = (offsets // (H * W)) % (n_elements // (H * W))
    
    # Load channel parameters
    mean = tl.load(running_mean_ptr + channel_idx, mask=mask)
    var = tl.load(running_var_ptr + channel_idx, mask=mask) + eps  # add eps for stability
    weight = tl.load(weight_ptr + channel_idx, mask=mask)
    bias = tl.load(bias_ptr + channel_idx, mask=mask)
    
    # Batch normalization: normalize, scale, shift
    x_norm = (x_added - mean) / tl.sqrt(var)
    x_bn = x_norm * weight + bias
    
    # ReLU activation: tmp_6 = relu(tmp_5, inplace=True)
    x_relu = tl.maximum(x_bn, 0.0)
    tl.store(output_relu_ptr + offsets, x_relu, mask=mask)

@torch.fx.wrap  
def fused_bn_relu_add(running_mean, running_var, weight, bias, add_tensor, input_tensor):
    """Wrapper for the fused kernel"""
    # Get tensor shape info
    n_elements = input_tensor.numel()
    B, C, H, W = input_tensor.shape
    
    # Determine optimal block size and grid dimensions
    BLOCK_SIZE = 1024  # Adjust based on GPU architecture
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    output_added = torch.empty_like(input_tensor)  # tmp_4: result after addition
    output_relu = torch.empty_like(input_tensor)   # tmp_6: result after ReLU
    
    # Launch kernel
    fused_bn_relu_add_kernel[(num_programs,)](
        input_tensor,  # input_ptr: in_5
        add_tensor,    # add_ptr: in_4
        output_added,  # output_added_ptr: where to store tmp_4
        output_relu,   # output_relu_ptr: where to store tmp_6
        running_mean,  # running_mean_ptr: in_0
        running_var,   # running_var_ptr: in_1
        weight,        # weight_ptr: in_3
        bias,          # bias_ptr: in_2
        n_elements,    # total elements
        H,             # height dimension
        W,             # width dimension
        1e-05,         # epsilon
        BLOCK_SIZE
    )
    
    # Return both results as (tmp_4, tmp_6) as in the original pattern
    return output_added, output_relu

def replacement_func():
    """Returns the fused kernel function"""
    return fused_bn_relu_add