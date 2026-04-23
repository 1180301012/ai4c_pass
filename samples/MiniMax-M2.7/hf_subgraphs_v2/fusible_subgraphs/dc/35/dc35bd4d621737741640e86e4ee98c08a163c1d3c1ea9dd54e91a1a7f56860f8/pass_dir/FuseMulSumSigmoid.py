import torch
import triton
import triton.language as tl


@triton.jit
def fused_mul_sum_sigmoid_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    batch_size,
    height,
    width,
    channels,
    num_outputs,  # batch * height * width
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that processes multiple outputs per block.
    
    Grid: (num_outputs / BLOCK_SIZE,) - each block processes BLOCK_SIZE outputs
    Each thread processes one output position
    """
    # Calculate position index within block
    pid = tl.program_id(0)
    pos_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid positions
    mask = pos_idx < num_outputs
    
    # Calculate indices from flat position
    batch_idx = pos_idx // (height * width)
    hw_idx = pos_idx % (height * width)
    height_idx = hw_idx // width
    width_idx = hw_idx % width
    
    # Calculate output offset
    output_offset = batch_idx * height * width + height_idx * width + width_idx
    
    # Calculate base offset for this position
    base_offset = batch_idx * channels * height * width + height_idx * width + width_idx
    
    # Initialize accumulator
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process all channels
    for ch in range(channels):
        offset = base_offset + ch * height * width
        x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
        y_val = tl.load(y_ptr + offset, mask=mask, other=0.0)
        sum_acc += x_val * y_val
    
    # Apply sigmoid
    output = 1.0 / (1.0 + tl.exp(-sum_acc))
    
    # Store result
    tl.store(output_ptr + output_offset, output, mask=mask)


@torch.fx.wrap
def fused_mul_sum_sigmoid(x, y):
    """
    Wrapper for the fused multiplication-sum-sigmoid kernel.
    
    Args:
        x: Input tensor of shape [batch, channels, height, width]
        y: Input tensor of shape [batch, channels, height, width]
    
    Returns:
        Output tensor of shape [batch, 1, height, width] after sigmoid(sum(x*y, dim=1)).unsqueeze(1)
    """
    batch_size, channels, height, width = x.shape
    nspatial = height * width
    num_outputs = batch_size * nspatial
    
    # Output shape: [batch, height, width] (before unsqueeze)
    output = torch.empty(batch_size, height, width, dtype=x.dtype, device=x.device)
    
    # Choose BLOCK_SIZE to balance parallelism and efficiency
    # For 4096 positions (batch=1): 64 threads * 64 iterations = 4096 parallelism
    # For 98304 positions (batch=24): 64 threads * 64 iterations = 6144 parallelism per block
    BLOCK_SIZE = 64
    
    # Calculate number of blocks needed
    num_blocks = (num_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_mul_sum_sigmoid_kernel[(num_blocks,)](
        x,
        y,
        output,
        batch_size,
        height,
        width,
        channels,
        num_outputs,
        BLOCK_SIZE,
    )
    
    # Expand to [batch, 1, height, width] to match unsqueeze(1) behavior
    return output.view(batch_size, 1, height, width)


def pattern(in_0, in_1):
    """
    Match the computation pattern:
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3
    """
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_mul_sum_sigmoid