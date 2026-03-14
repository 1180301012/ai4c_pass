import torch
import triton
import triton.language as tl


def pattern(input, weight, bias):
    """
    Match conv2d operation
    """
    result = torch.conv2d(input, weight, bias, (1, 1), (1, 1), (1, 1), 1)
    return result


def replacement_args(input, weight, bias):
    """
    Extract arguments
    """
    return (input, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['batch', 'out_channels', 'out_h', 'out_w'],
)
@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, in_h, in_w,
    out_channels, kernel_h, kernel_w,
    out_h, out_w,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    padding_h: tl.constexpr, padding_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized conv2d kernel with better memory access
    """
    pid = tl.program_id(0)
    total_elements = batch * out_channels * out_h * out_w
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decompose index
    ow = idx % out_w
    oh = (idx // out_w) % out_h
    oc = (idx // (out_w * out_h)) % out_channels
    b = idx // (out_w * out_h * out_channels)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load bias once per output channel
    bias_val = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    acc += bias_val
    
    # Compute convolution - unrolled for 3x3 kernel
    # Loop over input channels
    for ic in range(in_channels):
        # Unroll kernel loops for 3x3
        ih0 = oh * stride_h + 0 - padding_h
        iw0 = ow * stride_w + 0 - padding_w
        valid0 = (ih0 >= 0) & (ih0 < in_h) & (iw0 >= 0) & (iw0 < in_w) & mask
        input_idx0 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih0 * in_w + iw0
        weight_idx0 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 0
        input_val0 = tl.load(input_ptr + input_idx0, mask=valid0, other=0.0)
        weight_val0 = tl.load(weight_ptr + weight_idx0, mask=mask, other=0.0)
        acc += input_val0 * weight_val0
        
        ih0 = oh * stride_h + 0 - padding_h
        iw1 = ow * stride_w + 1 - padding_w
        valid1 = (ih0 >= 0) & (ih0 < in_h) & (iw1 >= 0) & (iw1 < in_w) & mask
        input_idx1 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih0 * in_w + iw1
        weight_idx1 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 1
        input_val1 = tl.load(input_ptr + input_idx1, mask=valid1, other=0.0)
        weight_val1 = tl.load(weight_ptr + weight_idx1, mask=mask, other=0.0)
        acc += input_val1 * weight_val1
        
        ih0 = oh * stride_h + 0 - padding_h
        iw2 = ow * stride_w + 2 - padding_w
        valid2 = (ih0 >= 0) & (ih0 < in_h) & (iw2 >= 0) & (iw2 < in_w) & mask
        input_idx2 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih0 * in_w + iw2
        weight_idx2 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 2
        input_val2 = tl.load(input_ptr + input_idx2, mask=valid2, other=0.0)
        weight_val2 = tl.load(weight_ptr + weight_idx2, mask=mask, other=0.0)
        acc += input_val2 * weight_val2
        
        ih1 = oh * stride_h + 1 - padding_h
        iw0 = ow * stride_w + 0 - padding_w
        valid3 = (ih1 >= 0) & (ih1 < in_h) & (iw0 >= 0) & (iw0 < in_w) & mask
        input_idx3 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih1 * in_w + iw0
        weight_idx3 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 3
        input_val3 = tl.load(input_ptr + input_idx3, mask=valid3, other=0.0)
        weight_val3 = tl.load(weight_ptr + weight_idx3, mask=mask, other=0.0)
        acc += input_val3 * weight_val3
        
        ih1 = oh * stride_h + 1 - padding_h
        iw1 = ow * stride_w + 1 - padding_w
        valid4 = (ih1 >= 0) & (ih1 < in_h) & (iw1 >= 0) & (iw1 < in_w) & mask
        input_idx4 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih1 * in_w + iw1
        weight_idx4 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 4
        input_val4 = tl.load(input_ptr + input_idx4, mask=valid4, other=0.0)
        weight_val4 = tl.load(weight_ptr + weight_idx4, mask=mask, other=0.0)
        acc += input_val4 * weight_val4
        
        ih1 = oh * stride_h + 1 - padding_h
        iw2 = ow * stride_w + 2 - padding_w
        valid5 = (ih1 >= 0) & (ih1 < in_h) & (iw2 >= 0) & (iw2 < in_w) & mask
        input_idx5 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih1 * in_w + iw2
        weight_idx5 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 5
        input_val5 = tl.load(input_ptr + input_idx5, mask=valid5, other=0.0)
        weight_val5 = tl.load(weight_ptr + weight_idx5, mask=mask, other=0.0)
        acc += input_val5 * weight_val5
        
        ih2 = oh * stride_h + 2 - padding_h
        iw0 = ow * stride_w + 0 - padding_w
        valid6 = (ih2 >= 0) & (ih2 < in_h) & (iw0 >= 0) & (iw0 < in_w) & mask
        input_idx6 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih2 * in_w + iw0
        weight_idx6 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 6
        input_val6 = tl.load(input_ptr + input_idx6, mask=valid6, other=0.0)
        weight_val6 = tl.load(weight_ptr + weight_idx6, mask=mask, other=0.0)
        acc += input_val6 * weight_val6
        
        ih2 = oh * stride_h + 2 - padding_h
        iw1 = ow * stride_w + 1 - padding_w
        valid7 = (ih2 >= 0) & (ih2 < in_h) & (iw1 >= 0) & (iw1 < in_w) & mask
        input_idx7 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih2 * in_w + iw1
        weight_idx7 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 7
        input_val7 = tl.load(input_ptr + input_idx7, mask=valid7, other=0.0)
        weight_val7 = tl.load(weight_ptr + weight_idx7, mask=mask, other=0.0)
        acc += input_val7 * weight_val7
        
        ih2 = oh * stride_h + 2 - padding_h
        iw2 = ow * stride_w + 2 - padding_w
        valid8 = (ih2 >= 0) & (ih2 < in_h) & (iw2 >= 0) & (iw2 < in_w) & mask
        input_idx8 = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih2 * in_w + iw2
        weight_idx8 = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + 8
        input_val8 = tl.load(input_ptr + input_idx8, mask=valid8, other=0.0)
        weight_val8 = tl.load(weight_ptr + weight_idx8, mask=mask, other=0.0)
        acc += input_val8 * weight_val8
    
    # Store output
    output_idx = b * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow
    tl.store(output_ptr + output_idx, acc, mask=mask)


@torch.fx.wrap
def triton_conv2d(input, weight, bias):
    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output size (stride=1, padding=1)
    out_h = in_h
    out_w = in_w
    
    output = torch.empty(
        (batch, out_channels, out_h, out_w),
        device=input.device,
        dtype=input.dtype
    )
    
    total_elements = batch * out_channels * out_h * out_w
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    optimized_conv2d_kernel[grid](
        input, weight, bias, output,
        batch, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w,
        1, 1,  # stride
        1, 1,  # padding
    )
    
    return output


def replacement_func():
    return triton_conv2d