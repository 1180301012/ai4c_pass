import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv1d_gelu_slice_kernel(
    # Inputs
    in_3_ptr, in_4_ptr, in_2_ptr,
    # Output
    tmp_7_out_ptr,
    # Shape info
    B, Cin, L_in,
    # Conv params
    stride_conv, padding_conv, dilation_conv, groups_conv, kernel_len,
    # Slicing info
    slice_channels,
    # Output length
    L_out,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    pid_out_time = tl.program_id(2)
    
    # Bounds check
    if pid_out_channel >= slice_channels or pid_out_time >= L_out:
        return
    
    # Grouped conv parameters
    # Weight shape: [768, 48, 31] = [groups * channels_per_group, channels_per_group, kernel_len]
    channels_per_group = Cin // groups_conv  # 48
    group_idx = pid_out_channel // channels_per_group
    
    # Compute convolution
    conv_acc = 0.0
    
    # Calculate input time for this output
    in_time_start = pid_out_time * stride_conv - padding_conv
    
    for ki in range(kernel_len):
        in_time = in_time_start + ki * dilation_conv
        if in_time >= 0 and in_time < L_in:
            # Input channel range for this group
            in_c_start = group_idx * channels_per_group
            for c in range(channels_per_group):
                # Input: [B, Cin, L_in] flattened as [B, Cin * L_in]
                in_offset = pid_batch * Cin * L_in + (in_c_start + c) * L_in + in_time
                val = tl.load(in_3_ptr + in_offset)
                
                # Weight: [Cout, channels_per_group, kernel_len]
                # w[pid_out_channel, c, ki]
                w_offset = pid_out_channel * channels_per_group * kernel_len + c * kernel_len + ki
                w = tl.load(in_4_ptr + w_offset)
                
                conv_acc = conv_acc + val * w
    
    # Add bias
    bias = tl.load(in_2_ptr + pid_out_channel)
    conv_acc = conv_acc + bias
    
    # GELU approximation: 0.5 * x * (1 + 0.044715 * x^3)
    c1 = 0.044715
    gelu_out = 0.5 * conv_acc * (1.0 + c1 * conv_acc * conv_acc * conv_acc)
    
    # Store sliced result
    out_offset = pid_batch * slice_channels * L_out + pid_out_channel * L_out + pid_out_time
    tl.store(tmp_7_out_ptr + out_offset, gelu_out)


@triton.jit
def fused_avg_pool_slice_kernel(
    # Input
    in_3_ptr,
    # Output
    tmp_6_out_ptr,
    # Shape info
    B, C, L_in,
    # Pool params
    kernel_size, stride_pool, padding_pool,
    # Slicing info
    slice_channels,
    # Output length
    L_out,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_out_time = tl.program_id(2)
    
    # Bounds check
    if pid_channel >= slice_channels or pid_out_time >= L_out:
        return
    
    # AvgPool computation
    in_time_start = pid_out_time * stride_pool - padding_pool
    
    pool_sum = 0.0
    count = 0
    
    for ki in range(kernel_size):
        in_time = in_time_start + ki
        if in_time >= 0 and in_time < L_in:
            offset = pid_batch * C * L_in + pid_channel * L_in + in_time
            val = tl.load(in_3_ptr + offset)
            pool_sum = pool_sum + val
            count = count + 1
    
    if count > 0:
        pool_out = pool_sum / tl.cast(count, tl.float32)
    else:
        pool_out = 0.0
    
    # Store sliced result
    out_offset = pid_batch * slice_channels * L_out + pid_channel * L_out + pid_out_time
    tl.store(tmp_6_out_ptr + out_offset, pool_out)


@triton.jit
def fused_add_transpose_layernorm_kernel(
    tmp_7_ptr, tmp_6_ptr,
    layernorm_weight_ptr, layernorm_bias_ptr,
    output_ptr,
    # Shape info
    B, C, L,
    # LayerNorm params
    eps,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    # Each block computes multiple output positions
    num_positions = B * L
    
    for idx in range(BLOCK_SIZE):
        pos = pid_block * BLOCK_SIZE + idx
        if pos < num_positions:
            b = pos // L
            s = pos % L
            
            # tmp_8 = tmp_6 + tmp_7 (both have shape [B, C, L])
            # tmp_9 = tmp_8.transpose(1, 2) -> [B, L, C]
            # LayerNorm over C dimension
            
            mean_acc = 0.0
            var_acc = 0.0
            
            # Compute mean and variance over C
            for c in range(C):
                t7_offset = b * C * L + c * L + s
                t6_offset = b * C * L + c * L + s
                t7 = tl.load(tmp_7_ptr + t7_offset)
                t6 = tl.load(tmp_6_ptr + t6_offset)
                val = t7 + t6
                mean_acc = mean_acc + val
                var_acc = var_acc + val * val
            
            mean_val = mean_acc / tl.cast(C, tl.float32)
            var_val = var_acc / tl.cast(C, tl.float32) - mean_val * mean_val
            inv_std = 1.0 / tl.sqrt(var_val + eps)
            
            # Compute normalized value
            sum_norm = 0.0
            for c in range(C):
                t7_offset = b * C * L + c * L + s
                t6_offset = b * C * L + c * L + s
                t7 = tl.load(tmp_7_ptr + t7_offset)
                t6 = tl.load(tmp_6_ptr + t6_offset)
                val = t7 + t6
                diff = val - mean_val
                weight = tl.load(layernorm_weight_ptr + c)
                sum_norm = sum_norm + diff * weight
            
            output_val = sum_norm * inv_std + tl.load(layernorm_bias_ptr + s)
            
            out_offset = b * L * C + s * C
            tl.store(output_ptr + out_offset, output_val)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3, in_4):
    """
    Fused kernel: conv1d + gelu + avg_pool1d + slice + add + transpose + layer_norm
    
    Input shapes:
    - in_0 (bias for layer_norm): [768]
    - in_1 (weight for layer_norm): [768]
    - in_2 (conv1d bias): [768]
    - in_3 (input): [1, 768, 249]
    - in_4 (conv1d weight): [768, 48, 31]
    
    Output: [1, 124, 768] (after transpose)
    """
    B, Cin, L_in = in_3.shape  # [1, 768, 249]
    
    # Conv1d params
    stride_conv = 2
    padding_conv = 15
    dilation_conv = 1
    groups_conv = 16
    kernel_len = 31
    
    # AvgPool1d params
    kernel_size = 2
    stride_pool = 2
    padding_pool = 0
    
    slice_channels = 124
    
    # Calculate output sizes
    L_out_conv = (L_in + 2 * padding_conv - dilation_conv * (kernel_len - 1) - 1) // stride_conv + 1  # = 124
    L_out_pool = (L_in - kernel_size) // stride_pool + 1  # = 124
    
    # Create output buffers for sliced outputs
    tmp_7_out = torch.empty((B, slice_channels, L_out_conv), dtype=in_3.dtype, device=in_3.device)
    tmp_6_out = torch.empty((B, slice_channels, L_out_pool), dtype=in_3.dtype, device=in_3.device)
    
    # Launch conv + gelu + slice kernel
    # Grid: [batch, out_channels, time_out]
    grid_conv = (B, slice_channels, L_out_conv)
    fused_conv1d_gelu_slice_kernel[grid_conv](
        in_3, in_4, in_2,
        tmp_7_out,
        B, Cin, L_in,
        stride_conv, padding_conv, dilation_conv, groups_conv, kernel_len,
        slice_channels,
        L_out_conv,
    )
    
    # Launch avg_pool + slice kernel
    # Grid: [batch, channels, time_out]
    grid_pool = (B, slice_channels, L_out_pool)
    fused_avg_pool_slice_kernel[grid_pool](
        in_3,
        tmp_6_out,
        B, Cin, L_in,
        kernel_size, stride_pool, padding_pool,
        slice_channels,
        L_out_pool,
    )
    
    # Compute add + transpose + layer_norm using Triton kernel
    L_out = L_out_conv
    C = slice_channels
    output = torch.empty((B, L_out, C), dtype=in_3.dtype, device=in_3.device)
    
    BLOCK_SIZE = 64
    num_blocks = (B * L_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_transpose_layernorm_kernel[(num_blocks,)](
        tmp_7_out, tmp_6_out,
        in_1, in_0,
        output,
        B, C, L_out,
        1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the full computation pattern:
    conv1d -> gelu -> slice +
    avg_pool1d -> slice -> add -> transpose -> layer_norm -> dropout
    """
    conv1d = torch.conv1d(in_3, in_4, in_2, (2,), (15,), (1,), 16)
    tmp_4 = torch.nn.functional.gelu(conv1d)
    tmp_5 = torch.nn.functional.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
    tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return tmp_11


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_kernel_wrapper