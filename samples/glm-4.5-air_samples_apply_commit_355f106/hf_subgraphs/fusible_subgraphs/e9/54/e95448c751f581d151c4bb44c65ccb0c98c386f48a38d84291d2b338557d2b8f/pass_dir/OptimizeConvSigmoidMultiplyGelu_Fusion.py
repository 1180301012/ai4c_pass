import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    return tmp_2, tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def conv_sigmoid_multiply_gelu_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    multiplier_ptr,
    conv_out_ptr,
    gelu_out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Determine program ID for batch processing
    pid = tl.program_id(0)
    
    # Only process within valid batch size
    if pid >= batch_size:
        return
    
    # Calculate base pointers for current batch element
    bias_base = bias_ptr + pid * out_channels
    weight_base = weight_ptr + pid * out_channels * in_channels * 1 * 1
    input_base = input_ptr + pid * in_channels * 1 * 1
    multiplier_base = multiplier_ptr + pid * out_channels * height * width
    conv_out_base = conv_out_ptr + pid * out_channels * height * width
    gelu_out_base = gelu_out_ptr + pid * out_channels * height * width
    
    # Process spatial dimensions
    for h in range(height):
        for w in range(width):
            # Initialize conv output with bias
            conv_out_val = tl.load(bias_base + 0).to(tl.float32)
            
            # Process channels (1x1 conv)
            for c in range(out_channels):
                # Load weight and input
                weight_val = tl.load(weight_base + c * in_channels * 1 * 1 + 0).to(tl.float32)
                input_val = tl.load(input_base + c * 1 * 1 + 0).to(tl.float32)
                multiplier_val = tl.load(multiplier_base + c * height * width + h * width + w).to(tl.float32)
                
                # Apply 1x1 convolution: output = bias + weight * input
                conv_val = conv_out_val + weight_val * input_val
                
                # Apply sigmoid: sigma(x) = 1 / (1 + exp(-x))
                sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
                
                # Apply element-wise multiplication and GELU
                # GELU(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                mul_val = sigmoid_val * multiplier_val
                gelu_val = mul_val * 0.5 * (1.0 + tl.tanh(0.7978845608 * (mul_val + 0.044715 * mul_val * mul_val * mul_val)))
                
                # Store both results
                tl.store(conv_out_base + c * height * width + h * width + w, conv_val)
                tl.store(gelu_out_base + c * height * width + h * width + w, gelu_val)

@torch.fx.wrap
def conv_sigmoid_multiply_gelu_fusion(in_0, in_1, in_2, in_3):
    batch_size = in_2.shape[0] if len(in_2.shape) == 4 else 1
    in_channels = in_1.shape[1]
    out_channels = in_1.shape[0]
    height = in_2.shape[2] if len(in_2.shape) == 4 else 1
    width = in_2.shape[3] if len(in_2.shape) == 4 else 1
    
    # Determine output dimensions based on conv2d parameters
    out_height = height
    out_width = width
    
    # Create output tensors
    conv_out = torch.empty((batch_size, out_channels, out_height, out_width), dtype=torch.float32, device=in_2.device)
    gelu_out = torch.empty((batch_size, out_channels, out_height, out_width), dtype=torch.float32, device=in_2.device)
    
    # Set grid size
    grid = (batch_size,)
    
    # Launch kernel for both outputs
    conv_sigmoid_multiply_gelu_kernel[grid](
        in_0,  # bias
        in_1,  # weight  
        in_3,  # input to conv
        in_2,  # multiplier
        conv_out,  # conv output (for pattern compatibility)
        gelu_out,  # fused result
        batch_size,
        in_channels,
        out_channels,
        out_height,
        out_width,
        BLOCK_SIZE=256,
    )
    
    return conv_out, gelu_out

def replacement_func():
    return conv_sigmoid_multiply_gelu_fusion