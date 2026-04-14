import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation pattern with //16 divisor
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0 // 16
    tmp_2 = torch.sym_sum([1, tmp_1])
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)

def replacement_args(in_0, in_1):
    return (in_0, in_1, 16)

@triton.jit
def relu_and_mean_kernel(
    input_ptr,
    output_ptr,
    relu_output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Optimized Triton kernel for ReLU + Mean computation
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1) 
    pid_w = tl.program_id(2)
    
    channel_start = pid_c * BLOCK_SIZE_C
    channel_end = min(channel_start + BLOCK_SIZE_C, n_channels)
    
    c_off = tl.arange(0, BLOCK_SIZE_C)
    h_off = pid_h * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    w_off = pid_w * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_abs = channel_start + c_off
    h_abs = h_off[:, None]
    w_abs = w_off[None, :]
    
    h_mask = h_abs < height
    w_mask = w_abs < width
    spatial_mask = h_mask & w_mask
    
    # Channel block processing
    for c_offset in range(BLOCK_SIZE_C):
        c_idx = channel_start + c_offset
        if c_idx >= n_channels:
            continue
            
        # Calculate pointer offsets for this channel
        input_base = input_ptr + c_idx * height * width
        relu_base = relu_output_ptr + c_idx * height * width
        mean_output_ptr = output_ptr + c_idx
        
        # Process spatial positions for this channel
        for h_offset in range(0, height, BLOCK_SIZE_N):
            for w_offset in range(0, width, BLOCK_SIZE_N):
                # Create local spatial window
                h_local = h_offset + tl.arange(0, BLOCK_SIZE_N)
                w_local = w_offset + tl.arange(0, BLOCK_SIZE_N)
                
                h_local_abs = h_local[:, None]
                w_local_abs = w_local[None, :]
                
                h_mask_local = h_local_abs < height
                w_mask_local = w_local_abs < width
                spatial_mask_local = h_mask_local & w_mask_local
                
                # Load and process input data
                input_vals = tl.load(input_base + h_local_abs * width + w_local_abs, 
                                   mask=spatial_mask_local, other=0.0)
                relu_vals = tl.maximum(input_vals, 0.0)
                
                # Store ReLU output
                tl.store(relu_base + h_local_abs * width + w_local_abs, relu_vals, mask=spatial_mask_local)
                
                # Accumulate for mean computation
                local_sum = tl.sum(relu_vals)
                # Atomic add to avoid race conditions
                tl.atomic_add(mean_output_ptr, local_sum)
    
    # Compute final mean by dividing by spatial size
    # This is done after all threads finish, but we'll handle it in the wrapper

@torch.fx.wrap
def fused_relu_and_mean(in_0, in_1, divisor):
    # Handle the division and sym_sum (these are simple operations)
    division_result = in_0 // divisor
    
    # For sym_sum([1, division_result]), we just need to compute 1 + division_result
    if torch.is_tensor(in_0):
        sym_sum_result = 1 + division_result
    else:
        sym_sum_result = 1 + division_result
    
    # Main optimization: ReLU + mean computation using Triton
    n, c, h, w = in_1.shape
    
    if n == 1 and h > 1 and w > 1:  # Only optimize for 4D tensors with spatial dimensions
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_C = 64
        
        num_c_blocks = (c + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
        num_h_blocks = (h + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        num_w_blocks = (w + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        
        # Create output tensors
        relu_output = torch.empty_like(in_1)
        mean_output = torch.empty((1, c, 1, 1), dtype=in_1.dtype, device=in_1.device)
        
        # Initialize mean output to zero
        mean_output.fill_(0.0)
        
        # Launch Triton kernel
        grid = (num_c_blocks, num_h_blocks, num_w_blocks)
        relu_and_mean_kernel[grid](
            input_ptr=in_1,
            output_ptr=mean_output.squeeze(-1).squeeze(-1).data_ptr(),  # Flattened mean output
            relu_output_ptr=relu_output.data_ptr(),
            n_channels=c,
            height=h,
            width=w,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        
        # Finalize mean computation
        mean_output = mean_output.squeeze(-1).squeeze(-1) / (h * w)
        mean_output = mean_output.reshape((1, c, 1, 1))
        
        return relu_output, mean_output
    else:
        # Fallback to regular computation for unsupported shapes
        relu_output = torch.nn.functional.relu(in_1, inplace=False)
        mean_output = relu_output.mean((2, 3), keepdim=True)
        return (relu_output, mean_output)

def replacement_func():
    return fused_relu_and_mean