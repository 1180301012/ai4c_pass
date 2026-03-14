import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def autotuned_conv_hardsigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C_out, C_IN,
    BLOCK_SIZE: tl.constexpr,
    WARP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C_out
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape to [N, C_out]
    n = offsets // C_out
    c = offsets % C_out
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + c, mask=mask)
    
    # Load input and weights using shared memory for better performance
    # Each warp processes one output channel for one batch element
    c_in_lane = tl.arange(0, WARP_SIZE)
    
    # Load input using strides
    x_base_address = n * C_IN
    x_addresses = x_base_address + c_in_lane
    x_row = tl.load(x_ptr + x_addresses, mask=c_in_lane < C_IN)
    
    # Load weights using strides  
    weights_base_address = c * C_IN
    weights_addresses = weights_base_address + c_in_lane
    weights = tl.load(weight_ptr + weights_addresses, mask=c_in_lane < C_IN)
    
    # Vectorized convolution using warp-level parallelism
    conv_partial = tl.sum(x_row * weights, axis=0) + bias
    
    # HardSigmoid activation using vectorized operations
    hardsigmoid_out = tl.maximum(0.0, tl.minimum(1.0, conv_partial * 0.16666667 + 0.5))
    # Alternative: hardsigmoid_out = (conv_partial + 3.0) * 0.16666667 for values in [-3, 3], clamped
    
    # Store result
    tl.store(out_ptr + n * C_out + c, hardsigmoid_out, mask=mask)

@triton.jit
def autotuned_elementwise_kernel(
    feature_map_ptr, se_output_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # 2D grid for better memory coalescing
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    total_x = tl.cdiv(N * C, BLOCK_SIZE_X)
    total_y = tl.cdiv(H * W, BLOCK_SIZE_Y)
    
    x_offsets = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_offsets = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    x_mask = x_offsets < N * C
    y_mask = y_offsets < H * W
    
    # Process a tile of output
    x_in_tile = x_offsets[:, None]
    y_in_tile = y_offsets[None, :]
    
    flat_indices = x_in_tile * (H * W) + y_in_tile
    mask = (x_in_tile < N * C) & (y_in_tile < H * W)
    
    # Load feature map
    feature_map = tl.load(feature_map_ptr + flat_indices, mask=mask)
    
    # Load SE output (broadcasted from [N, C] to [N, C, H, W])
    se_indices = x_in_tile
    se_output = tl.load(se_output_ptr + se_indices, mask=x_in_tile < N * C)
    
    # Element-wise multiplication
    output = feature_map * se_output
    
    # Store result
    tl.store(output_ptr + flat_indices, output, mask=mask)

@triton.jit
def autotuned_avg_pool_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    total_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_c = tl.cdiv(C, BLOCK_SIZE_C)
    
    n_start = pid_n * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    
    n_end = tl.minimum(n_start + BLOCK_SIZE_N, N)
    c_end = tl.minimum(c_start + BLOCK_SIZE_C, C)
    
    # Initialize accumulation for this (n_batch, c) block
    sum_val = 0.0
    
    # Accumulate over spatial dimensions
    for h in range(H):
        for w in range(W):
            input_offset = n_start * C * H * W + c_start * H * W + h * W + w
            for n_local in range(n_start, n_end):
                for c_local in range(c_start, c_end):
                    current_offset = n_local * C * H * W + c_local * H * W + h * W + w
                    value = tl.load(input_ptr + current_offset, other=0.0)
                    if n_local == n_start and c_local == c_start:  # Only accumulate in the main thread
                        sum_val += value
    
    # Compute average and store
    if n_start < N and c_start < C:
        avg_pool = sum_val / (H * W)
        tl.store(output_ptr + n_start * C + c_start, avg_pool)

@torch.fx.wrap
def autotuned_conv_hardsigmoid_triton(x, weight, bias, N, C_out, C_IN):
    # Autotune configuration for optimal performance
    configs = [
        triton.Config({'BLOCK_SIZE': 128, 'WARP_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'WARP_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'WARP_SIZE': 32}, num_stages=3, num_warps=4),
    ]
    
    @triton.autotune(configs=configs, key=['N', 'C_OUT', 'C_IN'])
    @triton.jit
    def autotuned_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, N, C_OUT, C_IN, BLOCK_SIZE, WARP_SIZE):
        pid = tl.program_id(0)
        total_elements = N * C_OUT
        
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        n = offsets // C_OUT
        c = offsets % C_OUT
        
        bias = tl.load(bias_ptr + c, mask=mask)
        
        c_in_lane = tl.arange(0, WARP_SIZE)
        
        x_base_address = n * C_IN
        x_addresses = x_base_address + c_in_lane
        x_row = tl.load(x_ptr + x_addresses, mask=c_in_lane < C_IN)
        
        weights_base_address = c * C_IN
        weights_addresses = weights_base_address + c_in_lane
        weights = tl.load(weight_ptr + weights_addresses, mask=c_in_lane < C_IN)
        
        conv_partial = tl.sum(x_row * weights, axis=0) + bias
        
        hardsigmoid_out = (conv_partial + 3.0) * 0.16666667
        hardsigmoid_out = tl.where(hardsigmoid_out > 1.0, 1.0, hardsigmoid_out)
        hardsigmoid_out = tl.where(hardsigmoid_out < 0.0, 0.0, hardsigmoid_out)
        
        tl.store(out_ptr + n * C_OUT + c, hardsigmoid_out, mask=mask)
    
    total_elements = N * C_out
    grid = (total_elements + 127) // 128
    
    out = torch.empty((N, C_out), dtype=torch.float32, device=x.device)
    
    autotuned_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N, C_OUT=C_out, C_IN=C_IN,
    )
    
    return out

@torch.fx.wrap
def autotuned_attention_multiply_triton(feature_map, se_output, N, C, H, W):
    # Configure grid and block sizes based on tensor dimensions
    total_elements = N * C * H * W
    grid_x = (N * C + 127) // 128
    grid_y = (H * W + 127) // 128
    
    out = torch.empty((N, C, H, W), dtype=torch.float32, device=feature_map.device)
    
    autotuned_elementwise_kernel[(grid_x, grid_y)](
        feature_map_ptr=feature_map,
        se_output_ptr=se_output,
        output_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_X=128,
        BLOCK_SIZE_Y=128
    )
    
    return out

@torch.fx.wrap
def autotuned_avg_pool_triton(input_tensor, N, C, H, W):
    grid_n = (N + 31) // 32
    grid_c = (C + 31) // 32
    
    out = torch.empty((N, C), dtype=torch.float32, device=input_tensor.device)
    
    autotuned_avg_pool_kernel[(grid_n, grid_c)](
        input_ptr=input_tensor,
        output_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_C=32
    )
    
    return out

def replacement_func():
    def optimized_forward(in_0, in_1, in_2, in_3):
        # Get input shapes
        in_1_shape = in_1.shape
        in_3_shape = in_3.shape
        in_2_shape = in_2.shape
        
        # Step 1: Autotuned Conv2D + HardSigmoid fusion
        se_weights = in_1.reshape(in_1_shape[0], in_1_shape[1])
        
        se_output = autotuned_conv_hardsigmoid_triton(
            x=in_3.reshape(in_3_shape[0], in_3_shape[1]),
            weight=se_weights,
            bias=in_0,
            N=in_3_shape[0],
            C_out=in_1_shape[0],
            C_IN=in_1_shape[1]
        )
        
        # Reshape for broadcasting
        se_output_reshaped = se_output.reshape(in_3_shape[0], in_1_shape[0], 1, 1)
        
        # Step 2: Autotuned element-wise multiplication
        multiplied = autotuned_attention_multiply_triton(
            feature_map=in_2,
            se_output=se_output_reshaped,
            N=in_2_shape[0],
            C=in_2_shape[1],
            H=in_2_shape[2],
            W=in_2_shape[3]
        )
        
        # Step 3: Autotuned average pooling + flatten
        pooled_flattened = autotuned_avg_pool_triton(
            input_tensor=multiplied,
            N=in_2_shape[0],
            C=in_2_shape[1],
            H=in_2_shape[2],
            W=in_2_shape[3]
        )
        
        # Return directly - dropout with 0.0 rate is identity operation
        return (pooled_flattened,)
    
    return optimized_forward