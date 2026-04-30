import torch
import triton
import triton.language as tl


@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Sizes
    batch, in_ch, in_h, in_w,
    out_ch,
    # Strides for input (row-major [b, c, h, w])
    in_batch_stride, in_ch_stride, in_h_stride, in_w_stride,
    # Stride for weight: weight[oc][k] -> oc * in_ch + k
    w_ch_stride,
    # Stride for output
    out_batch_stride, out_ch_stride, out_h_stride, out_w_stride,
    BLOCK_M: tl.constexpr,
):
    """
    1x1 Conv2d kernel using Triton.
    Each program handles one output spatial position.
    """
    pid = tl.program_id(0)
    
    # Calculate spatial indices
    batch_idx = pid // (in_h * in_w)
    spatial_idx = pid % (in_h * in_w)
    h_idx = spatial_idx // in_w
    w_idx = spatial_idx % in_w
    
    # For each output channel
    offs_oc = tl.arange(0, BLOCK_M)
    mask_oc = offs_oc < out_ch
    
    # Load bias
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)
    
    # Accumulator for all output channels
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Input position offset (scalar)
    in_offset_base = (batch_idx * in_batch_stride + 
                      h_idx * in_h_stride + 
                      w_idx * in_w_stride)
    
    # Iterate over input channels
    for k in range(in_ch):
        # Load weight for this channel: weight[oc, k]
        w_offsets = offs_oc * w_ch_stride + k
        w_vals = tl.load(weight_ptr + w_offsets, mask=mask_oc, other=0.0).to(tl.float32)
        
        # Load input value at this channel (scalar load, no mask needed)
        in_offset = in_offset_base + k * in_ch_stride
        in_val = tl.load(input_ptr + in_offset).to(tl.float32)
        
        # Multiply and accumulate: acc[oc] += w[oc, k] * in[k]
        acc += w_vals * in_val
    
    # Add bias and store
    result = (acc + bias).to(tl.float16)
    
    out_offset = (batch_idx * out_batch_stride + 
                  h_idx * out_h_stride + 
                  w_idx * out_w_stride +
                  offs_oc * out_ch_stride)
    tl.store(output_ptr + out_offset, result, mask=mask_oc)


def pattern(a, b, c):
    return torch.conv2d(a, b, c, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(a, b, c):
    return (a, b, c)


@torch.fx.wrap
def optimized_conv(x, weight, bias):
    """Optimized 1x1 Conv2d using Triton."""
    batch, in_ch, in_h, in_w = x.shape
    out_ch = weight.shape[0]
    
    # Allocate output
    output = torch.empty((batch, out_ch, in_h, in_w), 
                        dtype=x.dtype, device=x.device)
    
    # Input strides (row-major [b, c, h, w])
    in_batch_stride = in_ch * in_h * in_w
    in_ch_stride = in_h * in_w
    in_h_stride = in_w
    in_w_stride = 1
    
    # Weight stride: weight[oc][k] = oc * in_ch + k
    w_ch_stride = in_ch
    
    # Output strides (row-major [b, c, h, w])
    out_batch_stride = out_ch * in_h * in_w
    out_ch_stride = in_h * in_w
    out_h_stride = in_w
    out_w_stride = 1
    
    # Grid: one program per output spatial position
    num_programs = batch * in_h * in_w
    
    # Block size for output channels
    BLOCK_M = 128
    
    conv1x1_kernel[(num_programs,)](
        x, weight, bias, output,
        batch, in_ch, in_h, in_w,
        out_ch,
        in_batch_stride, in_ch_stride, in_h_stride, in_w_stride,
        w_ch_stride,
        out_batch_stride, out_ch_stride, out_h_stride, out_w_stride,
        BLOCK_M,
    )
    
    return output


def replacement_func():
    return optimized_conv