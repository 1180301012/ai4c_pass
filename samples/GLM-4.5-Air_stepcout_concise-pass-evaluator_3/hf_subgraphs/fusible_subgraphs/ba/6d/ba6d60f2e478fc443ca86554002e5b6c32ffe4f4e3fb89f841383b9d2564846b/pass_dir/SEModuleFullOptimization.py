import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # This function MUST exactly mirror the operations in model.py
    # Full computational pattern from the models:
    # Conv2D -> Sigmoid -> Element-wise mul -> ReLU -> BatchNorm
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_4
    tmp_5 = in_5
    tmp_6 = torch.conv2d(in_7, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
    tmp_7 = tmp_6.sigmoid()
    tmp_8 = in_6 * tmp_7
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
    tmp_10 = torch.nn.functional.batch_norm(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return (tmp_9, tmp_10)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)

@triton.jit
def conv2d_sigmoid_relu_kernel(
    x_ptr,  # in_7 conv input [N, C_in, H, W]
    conv_weight_ptr,  # tmp_5 [C_out, C_in, 1, 1]
    conv_bias_ptr,  # tmp_4 [C_out]
    gate_ptr,  # in_6 [N, C_out, H, W]
    sigmoid_out_ptr,  # output sigmoid
    relu_out_ptr,  # output relu
    N, C_in, C_out, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Program id
    pid = tl.program_id(0)
    
    # Calculate range for this program
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C_out * H * W
    
    # Load convolution result
    conv_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    # Load gate values and multiply
    gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    multiply_val = gate_val * sigmoid_val
    
    # Apply ReLU
    relu_val = tl.maximum(multiply_val, 0.0)
    
    # Store intermediate results
    tl.store(sigmoid_out_ptr + offsets, sigmoid_val, mask=mask)
    tl.store(relu_out_ptr + offsets, relu_val, mask=mask)

@triton.jit  
def batch_norm_kernel(
    x_ptr,  # input to batch_norm [N, C, H, W]
    running_mean_ptr,  # in_0 [C]
    running_var_ptr,  # in_1 [C] 
    bn_weight_ptr,  # in_3 [C]
    bn_bias_ptr,  # in_2 [C]
    out_ptr,  # output [N, C, H, W]
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program id  
    pid = tl.program_id(0)
    
    # Calculate range
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Get channel index for each element
    channel_idx = offsets % C
    
    # Load batch norm parameters - need to mask by C
    valid_mask = channel_idx < C
    
    mean = tl.load(running_mean_ptr + channel_idx, mask=valid_mask, other=0.0)
    var = tl.load(running_var_ptr + channel_idx, mask=valid_mask, other=1.0)
    weight = tl.load(bn_weight_ptr + channel_idx, mask=valid_mask, other=1.0)
    bias = tl.load(bn_bias_ptr + channel_idx, mask=valid_mask, other=0.0)
    
    # Apply batch normalization formula
    # Use tl.sqrt instead of missing rsqrt
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (x - mean) * inv_std * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, normalized, mask=mask)

@torch.fx.wrap
def se_module_optimized(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # Get tensor shapes
    N, C_in, H, W = in_7.shape
    C_out, _, _, _ = in_5.shape
    
    # Step 1: Optimize conv2d + sigmoid + elementwise mul + relu fusion
    # Allocate intermediate outputs
    sigmoid_out = torch.empty((N, C_out, H, W), dtype=torch.float32, device=in_7.device)
    relu_out = torch.empty((N, C_out, H, W), dtype=torch.float32, device=in_7.device)
    
    # Set up and launch conv-sigmoid-relu kernel
    BLOCK_SIZE = 1024
    numel_conv = N * C_out * H * W
    grid_conv = (numel_conv + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv2d_sigmoid_relu_kernel[grid_conv](
        in_7,
        in_5,  # conv_weight
        in_4,  # conv_bias
        in_6,  # gate
        sigmoid_out,
        relu_out,
        N, C_in, C_out, H, W,
        BLOCK_SIZE
    )
    
    # Step 2: Optimize batch normalization
    batch_norm_out = torch.empty_like(relu_out)
    numel_bn = N * C_out * H * W
    grid_bn = (numel_bn + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    batch_norm_kernel[grid_bn](
        relu_out,
        in_0,  # running_mean
        in_1,  # running_var
        in_3,  # bn_weight
        in_2,  # bn_bias  
        batch_norm_out,
        N, C_out, H, W,
        1e-05,  # epsilon
        BLOCK_SIZE
    )
    
    # Return the same as the original pattern: (tmp_9, tmp_10) -> (relu_out, batch_norm_out)
    return (relu_out, batch_norm_out)

def replacement_func():
    return se_module_optimized