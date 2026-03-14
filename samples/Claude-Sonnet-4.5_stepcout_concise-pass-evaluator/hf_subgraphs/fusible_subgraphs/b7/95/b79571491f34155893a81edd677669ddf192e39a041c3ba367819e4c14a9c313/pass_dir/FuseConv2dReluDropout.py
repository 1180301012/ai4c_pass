import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern to match: conv2d -> relu -> dropout(p=0.0)
    """
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(tmp_2, inplace=True)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)


@triton.jit
def conv2d_relu_kernel(
    # Input pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Dimensions
    batch_size, in_channels, in_h, in_w,
    out_channels, kernel_h, kernel_w,
    out_h, out_w,
    # Strides
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_oc_stride, weight_ic_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    # Block sizes
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate which output element this program is responsible for
    total_spatial = batch_size * out_h * out_w
    num_oc_blocks = tl.cdiv(out_channels, BLOCK_SIZE_OC)
    
    spatial_idx = pid // num_oc_blocks
    oc_block_idx = pid % num_oc_blocks
    
    # Decompose spatial_idx into batch, out_h, out_w
    batch_idx = spatial_idx // (out_h * out_w)
    remaining = spatial_idx % (out_h * out_w)
    oh_idx = remaining // out_w
    ow_idx = remaining % out_w
    
    # Output channel range for this block
    oc_start = oc_block_idx * BLOCK_SIZE_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < out_channels
    
    # Accumulator for the convolution result
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
    
    # Convolution computation: iterate over input channels and kernel spatial dimensions
    for ic_block_start in range(0, in_channels, BLOCK_SIZE_IC):
        ic_offsets = ic_block_start + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < in_channels
        
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Input spatial position
                ih = oh_idx + kh
                iw = ow_idx + kw
                
                # Check bounds
                if ih < in_h and iw < in_w:
                    # Load input: [BLOCK_SIZE_IC]
                    input_idx = (batch_idx * input_batch_stride + 
                                ic_offsets * input_channel_stride + 
                                ih * input_h_stride + 
                                iw * input_w_stride)
                    input_val = tl.load(input_ptr + input_idx, mask=ic_mask, other=0.0)
                    
                    # Load weights: [BLOCK_SIZE_OC, BLOCK_SIZE_IC]
                    for ic_local in range(BLOCK_SIZE_IC):
                        if ic_block_start + ic_local < in_channels:
                            ic = ic_block_start + ic_local
                            weight_idx = (oc_offsets * weight_oc_stride + 
                                        ic * weight_ic_stride + 
                                        kh * weight_h_stride + 
                                        kw * weight_w_stride)
                            weight_val = tl.load(weight_ptr + weight_idx, mask=oc_mask, other=0.0)
                            
                            # Accumulate
                            acc += weight_val * input_val[ic_local]
    
    # Add bias
    bias_val = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
    acc += bias_val
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store output
    output_idx = (batch_idx * output_batch_stride + 
                  oc_offsets * output_channel_stride + 
                  oh_idx * output_h_stride + 
                  ow_idx * output_w_stride)
    tl.store(output_ptr + output_idx, acc, mask=oc_mask)


@torch.fx.wrap
def fused_conv2d_relu(input_tensor, weight, bias):
    # Get dimensions
    batch_size, in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions (stride=1, padding=0, dilation=1)
    out_h = in_h - kernel_h + 1
    out_w = in_w - kernel_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_h, out_w, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate strides
    input_strides = input_tensor.stride()
    weight_strides = weight.stride()
    output_strides = output.stride()
    
    # Grid configuration
    BLOCK_SIZE_OC = 32
    BLOCK_SIZE_IC = 32
    
    total_spatial = batch_size * out_h * out_w
    num_oc_blocks = (out_channels + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC
    grid = (total_spatial * num_oc_blocks,)
    
    # Launch kernel
    conv2d_relu_kernel[grid](
        input_tensor, weight, bias, output,
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
    )
    
    return output


def replacement_func():
    return fused_conv2d_relu