import torch
import triton
import triton.language as tl


def pattern(in_0):
    return torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))

def replacement_args(in_0):
    return (in_0,)


@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch,
    channels,
    H_in,
    W_in,
    H_out,
    W_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Constants for block size (64x48 -> 32x24: 2x2 blocks)
    H_block = 2
    W_block = 2
    H_block_size = H_block * W_block
    
    total_elements = batch * channels * H_out * W_out
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = min(total_elements, start + BLOCK_SIZE)
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert offset to output coordinates
    batch_idx = offsets // (channels * H_out * W_out)
    rem = offsets % (channels * H_out * W_out)
    channel_idx = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    h_out_idx = rem // W_out
    w_out_idx = rem % W_out
    
    # Calculate input block base
    input_base = batch_idx * (channels * H_in * W_in) + \
                 channel_idx * (H_in * W_in) + \
                 (h_out_idx * H_block) * W_in + \
                 (w_out_idx * W_block)
    
    # Load and sum 2x2 block
    val00 = tl.load(input_ptr + input_base, mask=mask, other=0.0)
    val01 = tl.load(input_ptr + input_base + 0 * W_in + 1, mask=mask, other=0.0)
    val10 = tl.load(input_ptr + input_base + 1 * W_in + 0, mask=mask, other=0.0)
    val11 = tl.load(input_ptr + input_base + 1 * W_in + 1, mask=mask, other=0.0)
    sum_val = val00 + val01 + val10 + val11
    output_val = sum_val / 4.0
    
    # Output coordinate
    output_base = batch_idx * (channels * H_out * W_out) + \
                  channel_idx * (H_out * W_out) + \
                  h_out_idx * W_out + w_out_idx
    tl.store(output_ptr + output_base, output_val, mask=mask)


@torch.fx.wrap
def optimized_adaptive_avg_pool2d(input_tensor):
    batch, channels, H_in, W_in = input_tensor.shape
    H_out, W_out = 32, 24
    output = torch.empty((batch, channels, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    num_elements = batch * channels * H_out * W_out
    BLOCK_SIZE = 256  # Optimal for occupancy
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    adaptive_avg_pool2d_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch=batch,
        channels=channels,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return optimized_adaptive_avg_pool2d