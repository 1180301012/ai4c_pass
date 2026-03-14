import torch
import triton
import triton.language as tl

# Pattern: 1x1 conv2d -> gelu(approximate='none') -> dropout(p=0.0)
# The dropout with p=0.0 is a no-op, so we fuse conv2d + gelu

def pattern(bias, weight, input_tensor):
    """
    Match 1x1 conv2d + gelu + dropout pattern.
    conv2d with stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    """
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    gelu_out = torch.nn.functional.gelu(conv_out, approximate='none')
    dropout_out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return dropout_out


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.jit
def conv1x1_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C_in, C_out, H, W,
    input_batch_stride,
    input_channel_stride,
    input_h_stride,
    input_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_h_stride,
    output_w_stride,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Each program handles BLOCK_SIZE output elements
    pid = tl.program_id(0)
    total_elements = N * C_out * H * W
    
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    # Calculate indices
    hw = H * W
    chw = C_out * hw
    
    n = offs // chw
    remainder = offs % chw
    c_out = remainder // hw
    remainder2 = remainder % hw
    h = remainder2 // W
    w = remainder2 % W
    
    # Load bias
    bias_val = tl.load(bias_ptr + c_out, mask=mask, other=0.0)
    
    # Accumulate over input channels
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for c_in_start in range(0, C_in, BLOCK_C_IN):
        c_in_offs = c_in_start + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_offs < C_in
        
        for i in tl.static_range(BLOCK_C_IN):
            c_in = c_in_start + i
            if c_in < C_in:
                # Load input at this position
                input_offset = n * input_batch_stride + c_in * input_channel_stride + h * input_h_stride + w * input_w_stride
                inp_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                
                # Load weight: [C_out, C_in, 1, 1]
                weight_offset = c_out * C_in + c_in
                wgt = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
                
                acc += inp_val * wgt
    
    # Add bias
    acc = acc + bias_val
    
    # Apply GELU
    sqrt2 = 1.4142135623730951
    result = acc * 0.5 * (1.0 + tl.math.erf(acc / sqrt2))
    
    # Store output
    output_offset = n * output_batch_stride + c_out * output_channel_stride + h * output_h_stride + w * output_w_stride
    tl.store(output_ptr + output_offset, result, mask=mask)


@torch.fx.wrap
def fused_conv1x1_gelu(bias, weight, input_tensor):
    N, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    
    output = torch.empty((N, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    total_elements = N * C_out * H * W
    
    BLOCK_SIZE = 256
    BLOCK_C_IN = 64
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Pass weight directly, it's already contiguous with shape [C_out, C_in, 1, 1]
    conv1x1_gelu_kernel[grid](
        input_tensor, weight, bias, output,
        N, C_in, C_out, H, W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C_IN=BLOCK_C_IN,
    )
    
    return output


def replacement_func():
    return fused_conv1x1_gelu