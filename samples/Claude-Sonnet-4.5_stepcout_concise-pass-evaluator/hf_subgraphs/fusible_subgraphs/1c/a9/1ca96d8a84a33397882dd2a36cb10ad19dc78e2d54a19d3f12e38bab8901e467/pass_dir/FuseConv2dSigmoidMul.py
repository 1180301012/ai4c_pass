import torch
import triton
import triton.language as tl

def pattern(in_6, tmp_1, tmp_0, in_5):
    """
    Pattern: Conv2D + Sigmoid + Multiply
    """
    tmp_2 = torch.conv2d(in_6, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = in_5 * tmp_3
    return tmp_4

def replacement_args(in_6, tmp_1, tmp_0, in_5):
    return (in_6, tmp_1, tmp_0, in_5)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['batch', 'in_channels', 'out_channels', 'out_h', 'out_w'],
)
@triton.jit
def conv2d_1x1_sigmoid_mul_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    batch, in_channels, out_channels, out_h, out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused 1x1 Conv2D + Sigmoid + Multiply
    Processes output elements in parallel
    """
    pid = tl.program_id(0)
    
    # Each program processes BLOCK_SIZE output elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total output elements
    hw_size = out_h * out_w
    total_out = batch * out_channels * hw_size
    mask = offsets < total_out
    
    # Decode indices: [b, oc, spatial]
    b_idx = offsets // (out_channels * hw_size)
    oc_idx = (offsets // hw_size) % out_channels
    
    # Compute 1x1 convolution - vectorized over input channels
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for ic in range(in_channels):
        # Input: [b, ic, 1, 1] -> scalar per batch
        input_idx = b_idx * in_channels + ic
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        
        # Weight: [oc, ic, 1, 1] -> scalar per output channel
        weight_idx = oc_idx * in_channels + ic
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        
        acc += input_val * weight_val
    
    # Add bias and apply sigmoid in one step
    bias_val = tl.load(bias_ptr + oc_idx, mask=mask, other=0.0)
    sigmoid_out = tl.sigmoid(acc + bias_val)
    
    # Multiply with other tensor and store
    other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, sigmoid_out * other_val, mask=mask)

@torch.fx.wrap
def fused_conv2d_sigmoid_mul(input, weight, bias, other):
    """
    Fused Conv2D (1x1) + Sigmoid + Multiply with broadcasting
    """
    batch, in_channels, in_h, in_w = input.shape
    out_channels = weight.shape[0]
    batch_other, _, out_h, out_w = other.shape
    
    # Allocate output
    output = torch.empty((batch_other, out_channels, out_h, out_w), device=input.device, dtype=input.dtype)
    
    # Launch kernel with autotuned configuration
    total_elements = batch_other * out_channels * out_h * out_w
    grid = lambda meta: ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    conv2d_1x1_sigmoid_mul_kernel[grid](
        input, weight, bias, other, output,
        batch, in_channels, out_channels, out_h, out_w,
    )
    
    return output

def replacement_func():
    return fused_conv2d_sigmoid_mul