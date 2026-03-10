import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def depthwise_conv_bn_relu_kernel(
    input_ptr, 
    weight_ptr, 
    mean_ptr, 
    var_ptr, 
    gamma_ptr, 
    beta_ptr,
    output_ptr,
    N,       # batch size (1)
    IC,      # input channels (512)
    IH,      # input height (70)
    IW,      # input width (70)
    OC,      # output channels (same as IC for depthwise)
    OH,      # output height (64)
    OW,      # output width (64),
    KH,      # kernel height (7)
    KW,      # kernel width (7),
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs and calculate grid
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output channel dimension
    pid_k = tl.program_id(2)  # spatial dimension
    
    # Calculate offsets
    batch_offset = pid_m * IC * IH * IW
    out_channel_offset = pid_n * OH * OW
    
    # Load parameters for this channel
    mean = tl.load(mean_ptr + pid_n)
    var = tl.load(var_ptr + pid_n)
    gamma = tl.load(gamma_ptr + pid_n) if gamma_ptr is not None else 1.0
    beta = tl.load(beta_ptr + pid_n) if beta_ptr is not None else 0.0
    
    # Pre-compute normalization constants
    std = tl.sqrt(var + eps)
    inv_std = 1.0 / std
    scale = gamma * inv_std
    bias_scaled = beta - mean * scale
    
    # Process spatial block
    spatial_block_size = BLOCK_SIZE_K
    spatial_blocks = ((OH * OW) + spatial_block_size - 1) // spatial_block_size
    
    # Create mask for output spatial region
    out_spatial_offset = pid_k * spatial_block_size
    out_h = out_spatial_offset // OW
    out_w = out_spatial_offset % OW
    
    if out_h < OH and out_w < OW:
        # Depthwise convolution: each input processes its corresponding output
        in_h = out_h + KH // 2  # Simplified - assuming stride=1, padding='same'
        in_w = out_w + KW // 2
        
        if in_h < IH and in_w < IW:
            # Load input value (depthwise: same channel)
            input_val = tl.load(input_ptr + batch_offset + pid_n * IH * IW + in_h * IW + in_w)
            
            # Load weight value (depthwise: 1x1 kernel weight)
            weight_val = tl.load(weight_ptr + pid_n * 1 * 1 + 0)
            
            # Apply depthwise convolution
            conv_val = input_val * weight_val
            
            # Apply batch normalization and ReLU
            bn_val = conv_val * scale + bias_scaled
            relu_val = tl.maximum(bn_val, 0.0)
            
            # Store result
            output_offset = pid_m * OC * OH * OW + pid_n * OH * OW + out_h * OW + out_w
            tl.store(output_ptr + output_offset, relu_val)

@torch.fx.wrap
def fused_depthwise_conv_bn_relu(input_tensor, conv_weight, running_mean, running_var, weight, bias):
    # Get tensor shapes
    N, IC, IH, IW = input_tensor.shape
    OC, KH, KW, _ = conv_weight.shape
    
    # For depthwise conv with groups=512, OC should equal IC
    assert OC == IC, "For depthwise convolution, output channels must equal input channels"
    
    # Calculate output dimensions (assuming stride=1, padding='same')
    OH = IH - KH + 1
    OW = IW - KW + 1
    
    # Create output tensor
    output = torch.empty((N, OC, OH, OW), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions
    blocks_m = N
    blocks_n = OC
    blocks_k = (OH * OW + 1023) // 1024  # Spatial blocks of 1024
    
    # Launch Triton kernel
    fused_depthwise_conv_bn_relu_kernel[(blocks_m, blocks_n, blocks_k)](
        input_tensor,
        conv_weight,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        N, IC, IH, IW, OC, OH, OW, KH, KW,
        1e-05,  # eps
        1,      # BLOCK_SIZE_M (batch)
        1,      # BLOCK_SIZE_N (channels)
        1024,   # BLOCK_SIZE_K (spatial)
    )
    
    return output

def replacement_func():
    return fused_depthwise_conv_bn_relu