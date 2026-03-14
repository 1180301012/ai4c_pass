import torch
import triton
import triton.language as tl

def pattern(in_5):
    # Match avg_pool2d pattern
    tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None)
    return tmp_7

def replacement_args(in_5):
    return (in_5,)

@triton.jit
def optimized_avg_pool2d_kernel(
    input_ptr, output_ptr,
    N, C_in, H_in, W_in,
    pool_size, stride,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    batch_idx = pid // (C_in // BLOCK_SIZE_C)
    channel_idx = (pid % (C_in // BLOCK_SIZE_C)) * BLOCK_SIZE_C + tl.program_id(1)
    
    # Skip if out of bounds
    if batch_idx >= N or channel_idx >= C_in:
        return
    
    # Calculate output dimensions
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1
    
    # Process output tiles
    for h_out_base in range(0, H_out, BLOCK_SIZE_H):
        for w_out_base in range(0, W_out, BLOCK_SIZE_W):
            h_out_end = tl.minimum(h_out_base + BLOCK_SIZE_H, H_out)
            w_out_end = tl.minimum(w_out_base + BLOCK_SIZE_W, W_out)
            
            # Initialize accumulator for this tile
            tile_sum = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
            tile_count = 0
            
            # Accumulate over pooling window
            for ph in range(pool_size):
                for pw in range(pool_size):
                    h_in_start = h_out_base * stride + ph
                    w_in_start = w_out_base * stride + pw
                    
                    if h_in_start < H_in and w_in_start < W_in:
                        # Load input value
                        input_offset = (batch_idx * C_in + channel_idx) * H_in * W_in + h_in_start * W_in + w_in_start
                        input_val = tl.load(input_ptr + input_offset)
                        
                        # Accumulate to tile
                        h_tile = h_out_base + ph - h_out_start if h_out_base == h_out_start else ph
                        w_tile = w_out_base + pw - w_out_base if w_out_base == w_out_base else pw
                        
                        if h_tile < BLOCK_SIZE_H and w_tile < BLOCK_SIZE_W:
                            tile_sum[h_tile, w_tile] += input_val
                            tile_count += 1
            
            # Store averaged values
            for h_out_idx in range(h_out_base, h_out_end):
                for w_out_idx in range(w_out_base, w_out_end):
                    h_tile = h_out_idx - h_out_base
                    w_tile = w_out_idx - w_out_base
                    
                    if tile_count > 0:
                        avg_val = tile_sum[h_tile, w_tile] / float(pool_size * pool_size)
                    else:
                        avg_val = 0.0
                    
                    output_offset = (batch_idx * C_in + channel_idx) * H_out * W_out + h_out_idx * W_out + w_out_idx
                    tl.store(output_ptr + output_offset, avg_val)

@torch.fx.wrap
def optimized_avg_pool2d(input, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if divisor_override is not None:
        raise NotImplementedError("divisor_override not supported")
    
    N, C, H, W = input.shape
    
    # Handle count_include_pad and padding
    if count_include_pad and padding > 0:
        raise NotImplementedError("count_include_pad with padding not supported")
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)
    
    # Triton launch configuration
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    num_batches = (N * C + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_channels = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    optimized_avg_pool2d_kernel[(num_batches, num_channels)](
        input, output,
        N, C, H, W,
        kernel_size, stride,
        BLOCK_SIZE_N, BLOCK_SIZE_C,
        BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    return output

def replacement_func():
    return optimized_avg_pool2d