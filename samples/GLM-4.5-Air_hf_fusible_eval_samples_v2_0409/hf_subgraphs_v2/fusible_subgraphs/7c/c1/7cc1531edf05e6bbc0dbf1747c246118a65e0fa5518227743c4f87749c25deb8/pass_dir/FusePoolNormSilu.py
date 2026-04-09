import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    return tmp_5

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def triton_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr
):
    # Program id for parallelization
    pid = tl.program_id(0)
    
    # Calculate which spatial position this program handles
    spatial_programs = H // 2 * W // 2  # After pooling 
    spatial_pid = pid % spatial_programs
    channel_pid = pid // spatial_programs
    
    if channel_pid >= C:
        return
    
    # Calculate spatial coordinates after pooling (output is 8x8)
    out_h = (spatial_pid // (W // 2)) * 2  # Scale up for pooling input
    out_w = (spatial_pid % (W // 2)) * 2
    
    # Initialize pooling accumulator
    pool_sum = tl.zeros([], dtype=tl.float32)
    pool_count = 0
    
    # Perform 2x2 average pooling
    for dh in range(2):
        for dw in range(2):
            h_idx = out_h + dh
            w_idx = out_w + dw
            
            if h_idx < H and w_idx < W:
                # Calculate input tensor offset: [1, C, H, W]
                input_offset = channel_pid * H * W + h_idx * W + w_idx
                val = tl.load(input_ptr + input_offset)
                pool_sum += val
                pool_count += 1
    
    if pool_count > 0:
        pool_avg = pool_sum / tl.cast(pool_count, tl.float32)
    else:
        pool_avg = tl.zeros([], dtype=tl.float32)
    
    # Store output
    output_offset = channel_pid * (H // 2) * (W // 2) + spatial_pid
    tl.store(output_ptr + output_offset, pool_avg)

@torch.fx.wrap
def triton_avg_pool2d(input_tensor):
    # Get input dimensions - input should be [1, C, H, W]
    C, H, W = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
    N = input_tensor.shape[0]  # Should be 1
    
    # Output after pooling: [1, C, H/2, W/2]
    out_C = C
    out_H = H // 2
    out_W = W // 2
    out_elements = out_C * out_H * out_W
    
    # Create output tensor - result should be on same device as input
    output = torch.empty(out_C * out_H * out_W, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    triton_avg_pool2d_kernel[(grid_size,)](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.reshape(1, C, out_H, out_W)

def replacement_func():
    return triton_avg_pool2d