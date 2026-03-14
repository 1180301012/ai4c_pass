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
def optimised_conv_hardsigmoid_1x1_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C_out,
    C_IN: tl.constexpr,  # Make this a compile-time constant
):
    # Each program computes one output channel for one batch element
    pid = tl.program_id(0)
    
    # Compute n and c from program ID
    n = pid // C_out
    c = pid % C_out
    
    # Check if we're within bounds  
    if n >= N or c >= C_out:
        return
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + c)
    
    # Accumulate sum over input channels (C_IN is now compile-time constant)
    conv_sum = 0.0
    for c_in in range(C_IN):
        # Load input x[n, c_in]
        x_val = tl.load(x_ptr + n * C_IN + c_in)
        # Load weight[c, c_in]
        weight_val = tl.load(weight_ptr + c * C_IN + c_in)
        # Add to convolution sum
        conv_sum += weight_val * x_val
    
    conv_out = conv_sum + bias
    
    # HardSigmoid activation: clamp to [0, 1] range
    hardsigmoid_out = conv_out * 0.16666667 + 0.5
    hardsigmoid_out = tl.maximum(0.0, tl.minimum(1.0, hardsigmoid_out))
    
    # Store result at [n, c]
    tl.store(out_ptr + n * C_out + c, hardsigmoid_out)

@triton.jit
def optimised_elementwise_kernel(
    feature_map_ptr, se_output_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C * H * W, BLOCK_SIZE)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Load feature map element
    feature = tl.load(feature_map_ptr + offsets, mask=mask)
    
    # Load SE output (same for spatial positions) -> broadcast from [N, C] to [N, C, 1, 1]
    se_value = tl.load(se_output_ptr + offsets // (H * W), mask=mask)
    
    # Element-wise multiplication
    output = feature * se_value
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def optimised_avg_pool_flatten_kernel(
    input_ptr,
    output_ptr,
    N, C, 
    H: tl.constexpr, W: tl.constexpr,  # Make spatial dimensions compile-time constants
):
    # Each program processes one (n, c) pair
    pid = tl.program_id(0)
    
    # Compute n and c from program ID
    n = pid // C
    c = pid % C
    
    # Check if we're within bounds
    if n >= N or c >= C:
        return
    
    # Calculate sum over spatial dimensions
    spatial_sum = 0.0
    for h in range(H):
        for w in range(W):
            # Calculate global index
            global_index = n * C * H * W + c * H * W + h * W + w
            # Create mask for bounds checking
            mask = global_index < N * C * H * W
            # Load value
            value = tl.load(input_ptr + global_index, mask=mask)
            spatial_sum += value
    
    # Compute average pooling
    avg_pool = spatial_sum / (H * W)
    
    # Store result
    tl.store(output_ptr + n * C + c, avg_pool)

@torch.fx.wrap
def optimised_conv_hardsigmoid_se(x, weight, bias, batch_size, out_channels, in_C):
    # Determine grid size
    total_elements = batch_size * out_channels
    BLOCK_SIZE = 128  # Smaller block size for better occupancy in this kernel
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((batch_size, out_channels), dtype=torch.float32, device=x.device)
    
    optimised_conv_hardsigmoid_1x1_kernel[(batch_size * out_channels,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=batch_size,
        C_out=out_channels,
        C_IN=in_C
    )
    
    return out

@torch.fx.wrap
def optimised_attention_multiply(feature_map, se_output, N, C, H, W):
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024  # Larger block size for memory coalescing
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((N, C, H, W), dtype=torch.float32, device=feature_map.device)
    
    optimised_elementwise_kernel[(num_programs,)](
        feature_map_ptr=feature_map,
        se_output_ptr=se_output,
        output_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

@torch.fx.wrap
def optimised_avg_pool_flatten(input_tensor, N, C, H, W):
    total_elements = N * C
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((N, C), dtype=torch.float32, device=input_tensor.device)
    
    optimised_avg_pool_flatten_kernel[(N * C,)](
        input_ptr=input_tensor,
        output_ptr=out,
        N=N, C=C, 
        H=H, W=W  # These are now compile-time constants
    )
    
    return out

def replacement_func():
    def optimized_forward(in_0, in_1, in_2, in_3):
        # Get input shapes
        in_1_shape = in_1.shape
        in_3_shape = in_3.shape
        in_2_shape = in_2.shape
        
        # Step 1: Optimized 1x1 Conv2D + HardSigmoid fusion
        se_weights = in_1.reshape(in_1_shape[0], in_1_shape[1])  # [C_out, C_in]
        
        se_output = optimised_conv_hardsigmoid_se(
            x=in_3.reshape(in_3_shape[0], in_3_shape[1]),  # [N, C_in]
            weight=se_weights,
            bias=in_0,
            batch_size=in_3_shape[0],
            out_channels=in_1_shape[0],
            in_C=in_1_shape[1]  # C_in
        )
        
        # Reshape SE output back to [N, C_out, 1, 1] for broadcasting
        se_output_reshaped = se_output.reshape(in_3_shape[0], in_1_shape[0], 1, 1)
        
        # Step 2: Optimized element-wise multiplication
        multiplied = optimised_attention_multiply(
            feature_map=in_2,
            se_output=se_output_reshaped,
            N=in_2_shape[0],
            C=in_2_shape[1],
            H=in_2_shape[2],
            W=in_2_shape[3]
        )
        
        # Step 3: Optimized average pooling + flatten
        pooled_flattened = optimised_avg_pool_flatten(
            input_tensor=multiplied,
            N=in_2_shape[0],
            C=in_2_shape[1],
            H=in_2_shape[2],
            W=in_2_shape[3]
        )
        
        # Return directly - dropout with 0.0 rate is identity operation
        return (pooled_flattened,)
    
    return optimized_forward