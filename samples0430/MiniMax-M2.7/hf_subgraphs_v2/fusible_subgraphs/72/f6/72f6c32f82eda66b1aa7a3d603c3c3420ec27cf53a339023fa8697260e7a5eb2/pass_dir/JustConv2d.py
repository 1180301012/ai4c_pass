import torch
import triton
import triton.language as tl


# Autotune configuration for optimal performance
@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE': 32}, num_stages=2, num_warps=4),
        triton.Config({'TILE_SIZE': 64}, num_stages=2, num_warps=4),
        triton.Config({'TILE_SIZE': 128}, num_stages=2, num_warps=4),
    ],
    key=['in_channels'],
)
@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, output_ptr,
    # Input strides
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    # Weight strides
    weight_out_ch_stride, weight_in_ch_stride,
    # Output strides
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    # Shapes
    batch, in_channels, out_channels, height, width,
    TILE_SIZE: tl.constexpr,
    out_dtype: tl.constexpr,
):
    """Optimized Triton kernel for 1x1 conv2d with stride=1, padding=0, dilation=1
    
    Uses autotuning to find the best tile size.
    Each program handles one output element with efficient channel reduction.
    """
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Calculate output indices
    out_ch = pid // (height * width)
    spatial_idx = pid % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    batch_idx = (pid // (out_channels * height * width)) % batch
    
    # Compute output offset
    out_offset = (batch_idx * output_batch_stride + 
                  out_ch * output_channel_stride + 
                  h_idx * output_h_stride + 
                  w_idx * output_w_stride)
    
    # Accumulator - use vectorized operations
    acc = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    
    # Iterate over input channels in blocks - vectorized loading
    for c in range(0, in_channels, TILE_SIZE):
        offs = tl.arange(0, TILE_SIZE) + c
        mask = offs < in_channels
        
        # Vectorized load for input - all threads load different channels
        in_ptrs = input_ptr + (batch_idx * input_batch_stride + 
                               offs.to(tl.int64) * input_channel_stride + 
                               h_idx * input_h_stride + 
                               w_idx * input_w_stride)
        in_vals = tl.load(in_ptrs, mask=mask, other=0.0, cache_modifier='.cg')
        
        # Vectorized load for weights - all threads load different input channels
        w_ptrs = weight_ptr + (out_ch.to(tl.int64) * weight_out_ch_stride + 
                               offs.to(tl.int64) * weight_in_ch_stride)
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0, cache_modifier='.cg')
        
        # Fused multiply-add
        acc += in_vals.to(tl.float32) * w_vals.to(tl.float32)
    
    # Reduce and store
    result = tl.sum(acc, axis=0)
    tl.store(output_ptr + out_offset, result.to(out_dtype))


@torch.fx.wrap
def conv2d_1x1_wrapper(in_0, in_1):
    """Triton-based 1x1 conv2d wrapper"""
    batch, in_channels, height, width = in_1.shape
    out_channels = in_0.shape[0]
    dtype = in_1.dtype  # Use the same dtype as input
    
    # Allocate output with same dtype
    out = torch.empty((batch, out_channels, height, width), 
                      dtype=dtype, device=in_1.device)
    
    # Grid: one thread per output element
    total_outputs = batch * out_channels * height * width
    grid = (total_outputs,)
    
    # Convert dtype to Triton type
    if dtype == torch.float32:
        out_dtype = tl.float32
    elif dtype == torch.float16:
        out_dtype = tl.float16
    elif dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float16
    
    conv2d_1x1_kernel[grid](
        in_1, in_0, out,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, in_channels, out_channels, height, width,
        out_dtype=out_dtype,
    )
    
    return out


def pattern(in_0, in_1):
    """Match just conv2d"""
    return torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1):
    """Extract arguments"""
    return (in_0, in_1)


def replacement_func():
    """Return the replacement function"""
    return conv2d_1x1_wrapper