import torch
import triton
import triton.language as tl


@triton.jit
def se_block_fused_kernel(
    # Input pointers (raw pointers)
    x_ptr,
    se_ptr,
    weight_ptr,
    bias_ptr,
    # Output pointer
    out_ptr,
    # Shapes (passed as integers, not tensors)
    B: tl.constexpr,
    C: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    # Strides (passed as integers)
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    se_batch_stride,
    se_channel_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    out_batch_stride,
    out_channel_stride,
    out_height_stride,
    out_width_stride,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SE Block kernel: Conv2d + Sigmoid + Mul + Hardtanh
    
    Conv2d: se [B, G, 1, 1] @ weight [C, G, 1, 1] + bias [C] -> conv_out [B, C, 1, 1]
    Then: x * sigmoid(conv_out) -> hardtanh(..., 0, 6)
    """
    # Get program id for output element
    pid = tl.program_id(0)
    num_elements = B * C * H * W
    
    if pid * BLOCK_SIZE >= num_elements:
        return
    
    # Calculate linear index within bounds
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Decode linear index to (batch, channel, height, width)
    batch_idx = (offsets // (C * H * W)) % B
    channel_idx = (offsets // (H * W)) % C
    height_idx = (offsets // W) % H
    width_idx = offsets % W
    
    # Load main feature x [B, C, H, W]
    x_offset = (batch_idx * x_batch_stride + 
                channel_idx * x_channel_stride + 
                height_idx * x_height_stride + 
                width_idx * x_width_stride)
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    
    # === Compute SE block excitation ===
    # For each output channel c, compute: bias[c] + sum_g(se[b,g,0,0] * weight[c,g,0,0])
    # The conv is effectively a group convolution with groups = G where G = in_channels
    
    # Each thread computes one output channel's excitation
    excitation = tl.load(bias_ptr + channel_idx)
    
    # Iterate over input channels (group dimension)
    for g in range(G):
        # Load se features: se[b, g, 0, 0]
        se_offset = batch_idx * se_batch_stride + g * se_channel_stride
        se_val = tl.load(se_ptr + se_offset, mask=mask, other=0.0)
        
        # Load weight: weight[c, g, 0, 0]
        weight_offset = channel_idx * weight_out_channel_stride + g * weight_in_channel_stride
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Accumulate
        excitation = excitation + se_val * weight_val
    
    # Apply sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    # Cast to fp32 for exp computation (exp only supports fp32/fp64)
    excitation_fp32 = excitation.to(tl.float32)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-excitation_fp32))
    # Cast back to original type for multiplication
    sigmoid_val = sigmoid_val.to(x.dtype)
    
    # Compute: x * sigmoid(excitation)
    scaled = x * sigmoid_val
    
    # Apply Hardtanh: clamp to [0, 6]
    result = tl.minimum(tl.maximum(scaled, 0.0), 6.0)
    
    # Store result
    out_offset = (batch_idx * out_batch_stride + 
                  channel_idx * out_channel_stride + 
                  height_idx * out_height_stride + 
                  width_idx * out_width_stride)
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def se_block_fused(x, weight, bias, se):
    """
    Fused SE Block: Conv2d + Sigmoid + Mul + Hardtanh
    
    Args:
        x: Main feature tensor [B, C, H, W] on GPU
        weight: Conv weight [C, G, 1, 1] on CPU (will be loaded)
        bias: Conv bias [C] on CPU (will be loaded)
        se: SE features [B, G, 1, 1] on GPU
    """
    # Get shapes from the tensors directly
    x_shape = x.shape
    se_shape = se.shape
    weight_shape = weight.shape
    bias_shape = bias.shape
    
    B = x_shape[0]
    C = x_shape[1]
    H = x_shape[2]
    W = x_shape[3]
    G = se_shape[1]  # number of groups (reduction channels)
    
    # Allocate output using torch.empty_like (allowed operation)
    out = torch.empty_like(x)
    
    # Move weight and bias to GPU (allowed operation via .to())
    # Note: .to() is allowed as it's a tensor method, not a torch.* call
    weight_gpu = weight.to(x.device) if weight.device != x.device else weight
    bias_gpu = bias.to(x.device) if bias.device != x.device else bias
    
    # Calculate grid
    num_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Get strides
    x_strides = x.stride()
    se_strides = se.stride()
    out_strides = out.stride()
    
    # For weight [C, G, 1, 1], compute strides
    # weight_shape = [C, G, 1, 1]
    weight_strides = weight_gpu.stride()
    weight_out_stride = weight_strides[0]  # stride for C dimension
    weight_in_stride = weight_strides[1]   # stride for G dimension
    
    se_block_fused_kernel[(num_programs,)](
        # Input pointers
        x_ptr=x,
        se_ptr=se,
        weight_ptr=weight_gpu,
        bias_ptr=bias_gpu,
        # Output
        out_ptr=out,
        # Shapes
        B=B,
        C=C,
        G=G,
        H=H,
        W=W,
        # x strides
        x_batch_stride=x_strides[0],
        x_channel_stride=x_strides[1],
        x_height_stride=x_strides[2],
        x_width_stride=x_strides[3],
        # se strides
        se_batch_stride=se_strides[0],
        se_channel_stride=se_strides[1],
        # weight strides (flattened [C, G])
        weight_out_channel_stride=weight_out_stride,
        weight_in_channel_stride=weight_in_stride,
        # out strides
        out_batch_stride=out_strides[0],
        out_channel_stride=out_strides[1],
        out_height_stride=out_strides[2],
        out_width_stride=out_strides[3],
        # Block size
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the SE block pattern: Conv2d + Sigmoid + Mul + Hardtanh
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the replacement function.
    
    Pattern takes (in_0=bias, in_1=weight, in_2=x, in_3=se)
    Replacement expects (x, weight, bias, se)
    """
    return (in_2, in_1, in_0, in_3)


def replacement_func():
    """
    Return the replacement function.
    """
    return se_block_fused