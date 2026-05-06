import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, 672, None), slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_ptr,
    out_ptr,
    mean_ptr,
    B,
    N,
    H,
    W,
    stride_in,
    stride_out,
    stride_mean,
    dtype,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid layout: 1D grid per batch
    b_id = tl.program_id(0)
    
    # Process each block of channels
    r = tl.arange(0, BLOCK_SIZE)
    
    # Calculate the start index in channels
    c_start = tl.arange(0, BLOCK_SIZE) * N // BLOCK_SIZE
    
    # Process each channel chunk
    c = r + c_start
    
    # Mask to ensure we don't go out of bounds
    mask = c < N
    
    # Load values from input tensor
    # Load the input data
    in_values = tl.zeros((BLOCK_SIZE, H*W), dtype=dtype)
    for i in range(BLOCK_SIZE):
        if mask[i]:
            pos = b_id * stride_in + c[i] * stride_in + i * stride_in
            in_values[i] = tl.load(in_ptr + pos, mask=mask[i], other=0.0)    
    # Compute sum over spatial dimensions
    sum_values = tl.sum(in_values, axis=1)
    
    # Compute mean
    mean_values = sum_values / (H * W)
    
    # Store results
    tl.store(out_ptr + b_id * (B * N * H * W) + c * (H * W), in_values, mask=mask)
    tl.store(mean_ptr + b_id * N, mean_values, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    # Get input tensor details
    B = in_0.shape[0]
    N = 672
    H = in_0.shape[2]
    W = in_0.shape[3]
    
    # Allocate output tensors
    out = torch.empty_like(in_0)
    mean = torch.empty((B, N), dtype=in_0.dtype)
    
    # Launch kernel
    optimized_kernel[tl.cdiv(B, 1), tl.cdiv(N, 32)]( 
        in_ptr=in_0,
        out_ptr=out,
        mean_ptr=mean,
        B=B,
        N=N,
        H=H,
        W=W,
        stride_in=in_0.stride(0),
        stride_out=out.stride(0),
        stride_mean=mean.stride(0),
        dtype=in_0.dtype,
        BLOCK_SIZE=32
    )
    
    return (out, mean)

def replacement_func():
    return kernel_wrapper