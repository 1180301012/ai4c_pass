import torch
import triton
import triton.language as tl
import math

def pattern(in_3, in_1, in_0):
    # Conv2D operation from the original computation
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # View operation that reshapes the output - get batch size from input
    batch_size = in_3.shape[0]
    tmp_3 = conv2d.view(batch_size, 256, -1)
    return tmp_3

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def fused_conv2d_view_kernel(
    x_ptr,                    # Input tensor [B, 512, 64, 64]
    weight_ptr,              # Weights [256, 512, 1, 1]
    bias_ptr,                # Bias [256]
    out_ptr,                 # Output [B, 256, 4096]
    batch_size: tl.constexpr,
    n_channels_in: tl.constexpr,  # 512
    n_channels_out: tl.constexpr, # 256
    height: tl.constexpr,     # 64
    width: tl.constexpr,      # 64
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    dtype: tl.constexpr,
):
    # Calculate program ID and total programs
    pid_m = tl.program_id(0)
    
    # Ensure we don't go out of bounds for batch dimension
    if pid_m >= batch_size:
        return
    
    # Calculate output start position for this batch
    out_batch_start = pid_m * n_channels_out * (height * width)
    
    # Process each output channel
    for oc in range(0, n_channels_out, BLOCK_SIZE_N):
        oc_end = min(oc + BLOCK_SIZE_N, n_channels_out)
        
        # Load bias for this output channel slice
        bias = tl.load(bias_ptr + oc, mask=oc < n_channels_out, other=0.0)
        
        # Initialize accumulator for output slice
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=dtype)
        
        # Process input channels
        for ic in range(0, n_channels_in, BLOCK_SIZE_K):
            ic_end = min(ic + BLOCK_SIZE_K, n_channels_in)
            
            # Load weight for current output and input channel
            if oc < n_channels_out and ic < n_channels_in:
                weight = tl.load(weight_ptr + oc * n_channels_in + ic)
            else:
                weight = dtype(0.0)
            
            # Load input slice for current batch and process sum
            input_sum = dtype(0.0)
            for h in range(height):
                for w in range(width):
                    input_ptr = x_ptr + pid_m * n_channels_in * height * width + ic * height * width + h * width + w
                    if pid_m < batch_size and ic < n_channels_in:
                        val = tl.load(input_ptr, other=0.0)
                        input_sum += val
            
            # Accumulate weighted sum
            acc += weight * input_sum
        
        # Add bias for each spatial position
        for oc_off in range(oc, oc_end):
            final_out = acc[oc_off - oc] + bias
            # Store result for all spatial positions in the current output channel
            for h in range(height):
                for w in range(width):
                    output_pos = out_batch_start + oc_off * (height * width) + h * width + w
                    tl.store(out_ptr + output_pos, final_out)

@torch.fx.wrap
def fused_conv2d_view(in_3, in_1, in_0):
    B, C_in, H, W = in_3.shape
    C_out = in_0.shape[0]
    
    # Create output tensor with the desired shape [B, C_out, H*W]
    out = torch.empty((B, C_out, H * W), dtype=in_3.dtype, device=in_3.device)
    
    # Map PyTorch dtype to Triton dtype
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_dtype = dtype_map[in_3.dtype]
    
    # Tile sizes for better GPU utilization
    BLOCK_SIZE_M = 1      # Process one batch at a time
    BLOCK_SIZE_N = 64     # Number of output channels to process
    BLOCK_SIZE_K = 512    # Number of input channels to process
    
    # Calculate grid dimensions
    grid = (B,)
    
    # Launch the kernel
    fused_conv2d_view_kernel[grid](
        x_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=out,
        batch_size=B,
        n_channels_in=C_in,
        n_channels_out=C_out,
        height=H,
        width=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        dtype=triton_dtype,
    )
    
    return out

def replacement_func():
    return fused_conv2d_view