import torch
import triton
import triton.language as tl

@triton.jit
def fused_batch_norm_add_relu_kernel(
    in_4_ptr,
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    in_5_ptr,
    out_ptr,
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread index.
    b = tl.program_id(0) % batch
    c = tl.program_id(0) // batch
    block_h = tl.program_id(1)
    block_w = tl.program_id(2)
    
    thread_x = tl.thread_id(0) % BLOCK_SIZE
    thread_y = tl.thread_id(0) // BLOCK_SIZE
    
    h = block_h * BLOCK_SIZE + thread_x
    w = block_w * BLOCK_SIZE + thread_y
    
    mask = (h < height) & (w < width)
    
    # Load channel-specific parameters.
    mean_val = tl.load(in_0_ptr + c)
    var_val = tl.load(in_1_ptr + c)
    weight_val = tl.load(in_3_ptr + c)
    bias_val = tl.load(in_2_ptr + c)
    
    # Calculate the inverse standard deviation.
    inv_std = 1.0 / tl.sqrt(var_val + 1e-5)
    
    # Compute the data offset for the current channel and batch.
    data_offset = (b * channels + c) * height * width
    in_4_idx = data_offset + h * width + w
    in_5_idx = data_offset + h * width + w
    
    in_4_val = tl.load(in_4_ptr + in_4_idx, mask=mask)
    in_5_val = tl.load(in_5_ptr + in_5_idx, mask=mask)
    
    # Batch Norm + Add + ReLU.
    normalized = (in_4_val - mean_val) * inv_std * weight_val + bias_val
    summed = normalized + in_5_val
    out_val = tl.maximum(0.0, summed)
    
    tl.store(out_ptr + in_4_idx, out_val, mask=mask)

@torch.fx.wrap
def fused_batch_norm_add_relu(in_4, in_0, in_1, in_2, in_3, in_5):
    batch, channels, height, width = in_4.shape
    BLOCK_SIZE = 16
    grid = (batch * channels, (height + BLOCK_SIZE - 1) // BLOCK_SIZE, (width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    out = torch.empty_like(in_4)
    fused_batch_norm_add_relu_kernel[grid](
        in_4,
        in_0,
        in_1,
        in_2,
        in_3,
        in_5,
        out,
        batch,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def pattern(in_4, in_0, in_1, in_3, in_2, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace = False)
    return tmp_6

def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5)

def replacement_func():
    return fused_batch_norm_add_relu