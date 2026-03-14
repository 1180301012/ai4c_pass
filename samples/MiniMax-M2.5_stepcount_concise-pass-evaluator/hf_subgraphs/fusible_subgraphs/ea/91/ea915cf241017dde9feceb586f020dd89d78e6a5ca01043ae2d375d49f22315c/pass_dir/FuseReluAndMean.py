import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_0):
    """
    Match the pattern: relu(in_0) -> mean over (2, 3) -> view to (1, 1, -1)
    Returns both tmp_0 (relu output) and tmp_2 (reshaped mean)
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_2 = tmp_1.view(1, 1, -1)
    return tmp_0, tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def relu_mean_kernel(
    input_ptr,
    relu_output_ptr,
    mean_output_ptr,
    # Tensor dimensions
    batch_size,
    channels,
    height,
    width,
    # Strides
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    output_stride_b,
    output_stride_c,
    # Block size for spatial reduction
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Mean kernel.
    Each program handles one channel in one batch.
    Threads in the block cooperate to compute mean over spatial dimensions.
    """
    # Get channel index (we have channels * batch_size programs)
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Calculate base offset for this channel
    base_offset = batch_idx * input_stride_b + channel_idx * input_stride_c
    
    # Spatial size
    spatial_size = height * width
    
    # Each thread processes multiple elements using BLOCK_SIZE stride
    # Thread ID ranges from 0 to BLOCK_SIZE-1
    tid = tl.thread_id(0)
    
    # Initialize sum accumulator for this thread
    sum_acc = 0.0
    
    # Process all spatial positions with stride = BLOCK_SIZE
    # Each thread processes: tid, tid+BLOCK_SIZE, tid+2*BLOCK_SIZE, ...
    for spatial_idx in tl.range(tid, spatial_size, BLOCK_SIZE):
        # Convert linear index to 2D (h, w)
        h = spatial_idx // width
        w = spatial_idx % width
        
        # Calculate offset
        offset = base_offset + h * input_stride_h + w * input_stride_w
        
        # Load value
        val = tl.load(input_ptr + offset).to(tl.float32)
        
        # Apply ReLU and store to output
        relu_val = tl.where(val > 0, val, 0.0)
        tl.store(relu_output_ptr + offset, relu_val)
        
        # Accumulate sum
        sum_acc += relu_val
    
    # Parallel reduction using tl.reduce
    # Sum across all threads in the block
    sum_acc = tl.sum(sum_acc, axis=0)
    
    # Only thread 0 writes the final mean
    if tid == 0:
        mean_val = sum_acc / spatial_size
        out_offset = batch_idx * output_stride_b + channel_idx * output_stride_c
        tl.store(mean_output_ptr + out_offset, mean_val)


@torch.fx.wrap
def fused_relu_mean(x):
    """
    Fused ReLU + Mean over spatial dimensions (2, 3).
    Returns (relu_output, mean_reshaped)
    """
    batch, channels, height, width = x.shape
    spatial_size = height * width
    
    # Create output tensors
    relu_output = torch.empty_like(x)
    # Mean output shape: [batch, channels] -> reshape to [1, 1, channels]
    mean_output = torch.empty((batch, channels), dtype=x.dtype, device=x.device)
    
    # Choose BLOCK_SIZE based on spatial size
    if spatial_size >= 1024:
        BLOCK_SIZE = 1024
    elif spatial_size >= 512:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    # Grid: one program per batch * channel
    grid = (batch * channels,)
    
    relu_mean_kernel[grid](
        x,
        relu_output,
        mean_output,
        batch,
        channels,
        height,
        width,
        x.stride(0),  # input_stride_b
        x.stride(1),  # input_stride_c
        x.stride(2),  # input_stride_h
        x.stride(3),  # input_stride_w
        mean_output.stride(0),  # output_stride_b
        mean_output.stride(1),  # output_stride_c
        BLOCK_SIZE,
    )
    
    # Reshape mean output to [1, 1, channels]
    mean_reshaped = mean_output.view(1, 1, channels)
    
    return relu_output, mean_reshaped


def replacement_func():
    return fused_relu_mean