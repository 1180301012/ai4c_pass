import torch
import triton
import triton.language as tl


@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    # Strides for inputs [B, C, H, W]
    stride_in_0_b,
    stride_in_0_c,
    stride_in_0_h,
    stride_in_0_w,
    stride_in_1_b,
    stride_in_1_c,
    stride_in_1_h,
    stride_in_1_w,
    # Shape params
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: sigmoid((in_0 * in_1).sum(dim=1))
    
    Input shape: [batch, channels, height, width]
    Output shape: [batch, 1, height, width]
    
    Uses vectorized loads to process multiple channels per iteration.
    """
    # 1D grid: each program handles one output position (batch, h, w)
    pid = tl.program_id(0)
    
    # Decode pid to (batch_idx, h, w)
    batch_idx = pid // (height * width)
    remainder = pid % (height * width)
    h = remainder // width
    w = remainder % width
    
    # Calculate output offset - contiguous [B, 1, H, W]
    out_offset = batch_idx * (height * width) + h * width + w
    
    # Calculate base offsets for all channels
    in_0_base = batch_idx * stride_in_0_b + h * stride_in_0_h + w * stride_in_0_w
    in_1_base = batch_idx * stride_in_1_b + h * stride_in_1_h + w * stride_in_1_w
    
    # Accumulate sum using vectorized loads
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_val = 0.0
    
    # Process channels in blocks
    num_blocks = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_idx in range(num_blocks):
        c_offsets = block_idx * BLOCK_SIZE + offsets
        mask = c_offsets < channels
        
        in_0_offs = in_0_base + c_offsets * stride_in_0_c
        in_1_offs = in_1_base + c_offsets * stride_in_1_c
        
        x0 = tl.load(in_0_ptr + in_0_offs, mask=mask, other=0.0)
        x1 = tl.load(in_1_ptr + in_1_offs, mask=mask, other=0.0)
        
        prod = x0 * x1
        block_sum = tl.sum(prod, axis=0)
        sum_val = sum_val + block_sum
    
    # Compute sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-sum_val))
    
    # Store result
    tl.store(out_ptr + out_offset, sigmoid_val)


@torch.fx.wrap
def fused_mul_sum_sigmoid_wrapper(in_0, in_1):
    """
    Wrapper for the fused multiplication + sum + sigmoid operation.
    
    Args:
        in_0: Tensor of shape [batch, channels, height, width]
        in_1: Tensor of shape [batch, channels, height, width]
    
    Returns:
        Tensor of shape [batch, 1, height, width] = sigmoid((in_0 * in_1).sum(dim=1))
    """
    batch, channels, height, width = in_0.shape
    
    # Number of output elements = batch * height * width
    n_output_elements = batch * height * width
    
    # Allocate output for the sigmoid result
    out = torch.empty((batch, 1, height, width), dtype=in_0.dtype, device=in_0.device)
    
    # Get strides
    stride_in_0 = in_0.stride()
    stride_in_1 = in_1.stride()
    
    # 1D grid: one program per output element
    grid = (n_output_elements,)
    
    # Launch kernel to compute mul + sum + sigmoid
    # Use larger BLOCK_SIZE for better vectorization (channels=64 works well with 128)
    BLOCK_SIZE = 128 if channels <= 64 else 64
    
    fused_mul_sum_sigmoid_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        stride_in_0_b=stride_in_0[0],
        stride_in_0_c=stride_in_0[1],
        stride_in_0_h=stride_in_0[2],
        stride_in_0_w=stride_in_0[3],
        stride_in_1_b=stride_in_1[0],
        stride_in_1_c=stride_in_1[1],
        stride_in_1_h=stride_in_1[2],
        stride_in_1_w=stride_in_1[3],
        batch_size=batch,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match pattern: (in_1 * in_0).sum(dim=1).unsqueeze(1).sigmoid()
    """
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_mul_sum_sigmoid_wrapper