import torch
import triton
import triton.language as tl

# Pattern matching function (exactly mirrors the graph's operations)
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6

# Extract all required inputs

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused Conv2D + element-wise ops
@triton.jit
def fused_conv_kernel(
    in_ptr,  # Input tensor [batch, C_in, H, W]
    weight_ptr,  # Weight [C_out, C_in]
    bias_ptr,  # Bias [C_out]
    in2_ptr,  # In2 tensor [batch, C_out, H, W]
    out_ptr,  # Output tensor [batch, C_out, H, W]
    batch,  # Batch size
    C_in,  # Input channels
    C_out,  # Output channels
    H,  # Height
    W,  # Width
    BLOCK_SIZE: tl.constexpr = 32
):
    # Thread indices
    pid = tl.program_id(0)
    block_h = pid // (H * W)
    block_w = (pid % (H * W)) // H
    block_c = (pid % (H * W)) % H
    
    # Compute base offsets for this spatial block
    base_h = block_h * BLOCK_SIZE
    base_w = block_w * BLOCK_SIZE
    base_c = block_c * BLOCK_SIZE
    
    # Load input channels for current (h, w)
    # [batch, C_in] at (base_h, base_w)
    input_offsets = tl.arange(0, 1)[:, None] * (C_in * H * W) + \
                   tl.arange(0, 32)[None, :] * (H * W) + \
                   base_h * W + base_w
    input_data = tl.load(in_ptr + input_offsets, 
                        mask=(base_h + tl.arange(0, BLOCK_SIZE))[:, None] < H, 
                        other=0.0)
    
    # Load weight (cache all weights for this block)
    weight_offsets = tl.arange(0, 32)[:, None] * C_in + tl.arange(0, 32)[None, :]
    weight_data = tl.load(weight_ptr + weight_offsets, 
                         mask=tl.arange(0, 32)[:, None] < C_out,
                         other=0.0)
    
    # Compute convolution + element-wise ops
    output_data = tl.zeros((32, 32), dtype=tl.float32)
    output_data = tl.dot(input_data, weight_data.T)
    output_data += tl.load(bias_ptr)
    output_data = (output_data + 1.0) / 2.0
    output_data = tl.clamp(output_data, 0.0, 1.0)
    
    # Multiply by in2
    in2_offsets = tl.arange(0, 32)[:, None] * (C_out * H * W) + \
                  tl.arange(0, 32)[None, :] * (H * W) + \
                  base_h * W + base_w
    in2_data = tl.load(in2_ptr + in2_offsets, 
                      mask=(base_h + tl.arange(0, BLOCK_SIZE))[:, None] < H, 
                      other=0.0)
    output_data *= in2_data
    
    # Store output
    output_offsets = tl.arange(0, 32)[:, None] * (C_out * H * W) + \
                    tl.arange(0, 32)[None, :] * (H * W) + \
                    base_h * W + base_w
    tl.store(out_ptr + output_offsets, 
            output_data, 
            mask=(base_h + tl.arange(0, BLOCK_SIZE))[:, None] < H)

# Wrapper function for Triton kernel
@torch.fx.wrap
def fused_conv(in_0, in_1, in_2, in_3):
    # Extract tensor shapes
    batch = in_3.shape[0]
    C_in = in_3.shape[1]
    H = in_3.shape[2]
    W = in_3.shape[3]
    C_out = in_0.shape[0]
    
    # Create output tensor
    out = torch.empty((batch, C_out, H, W), dtype=in_3.dtype, device=in_3.device)
    
    # Configure Triton grid
    num_blocks = H * W * ((C_out + 31) // 32)
    
    # Launch kernel
    fused_conv_kernel[(num_blocks,)](
        in_3, in_1, in_0, in_2, out,
        batch, C_in, C_out, H, W,
        BLOCK_SIZE=32
    )
    
    return out

# Replacement function

def replacement_func():
    return fused_conv