import torch
import triton
import triton.language as tl

def pattern(in_5, tmp_3, tmp_2, tmp_1, tmp_0):
    # Conv2D operation
    tmp_5 = torch.conv2d(in_5, tmp_3, tmp_2, (2, 2), (1, 1), (1, 1), 1)
    # View operation
    tmp_6 = tmp_5.view(1, 384, 576)
    # Permute operation
    tmp_7 = tmp_6.permute(0, 2, 1)
    # LayerNorm operation
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (384,), tmp_1, tmp_0, 1e-05)
    return tmp_7, tmp_8

def replacement_args(in_5, tmp_3, tmp_2, tmp_1, tmp_0):
    return (in_5, tmp_3, tmp_2, tmp_1, tmp_0)

@triton.jit
def conv_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, 
    gamma_ptr, beta_ptr,
    intermediate_ptr, result_ptr,
    batch_size, in_channels, out_channels,
    in_height, in_width,
    BLOCK_HW: tl.constexpr
):
    # Program ID for parallel execution across spatial positions
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_height = (in_height + 2*1 - 1*3 - 1) // 2 + 1  # Same as original: 48 -> 24
    out_width = (in_width + 2*1 - 1*3 - 1) // 2 + 1
    
    # Each program handles one spatial position for all batch and channels
    total_positions = out_height * out_width
    position = pid % total_positions
    batch_idx = pid // total_positions
    
    if batch_idx >= batch_size or position >= total_positions:
        return
        
    h_out = position // out_width
    w_out = position % out_width
    
    # First compute convolution at this spatial position
    conv_values = tl.zeros(out_channels, dtype=tl.float32)
    
    for oc in range(out_channels):
        acc = 0.0
        for ic in range(in_channels):
            for kh in range(3):  # 3x3 kernel
                for kw in range(3):
                    h_in = h_out * 2 - 1 + kh  # stride=2, padding=1
                    w_in = w_out * 2 - 1 + kw
                    if 0 <= h_in < in_height and 0 <= w_in < in_width:
                        # Load input value
                        x_offset = batch_idx * in_channels * in_height * in_width + ic * in_height * in_width + h_in * in_width + w_in
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Load weight value
                        w_offset = oc * in_channels * 3 * 3 + ic * 3 * 3 + kh * 3 + kw
                        w_val = tl.load(weight_ptr + w_offset)
                        
                        acc += x_val * w_val
        
        # Add bias
        b_offset = oc
        bias_val = tl.load(bias_ptr + b_offset)
        conv_values[oc] = acc + bias_val
    
    # Store intermediate result (conv + bias) in permuted format [batch, channels, positions]
    # We need to map this to the final output format [batch, positions, channels] for the result
    pos_offset = batch_idx * out_channels * total_positions + position * out_channels
    for oc in range(out_channels):
        tl.store(intermediate_ptr + pos_offset + oc, conv_values[oc])
    
    # Store final result after layer norm (same format)
    # Layer norm normalization happens per position (not per channel!)
    # So we need to compute mean and variance across channels for this position
    
    # Compute mean across channels at this position
    mean = 0.0
    for oc in range(out_channels):
        mean += conv_values[oc]
    mean = mean / out_channels
    
    # Compute variance across channels at this position
    var = 0.0
    for oc in range(out_channels):
        diff = conv_values[oc] - mean
        var += diff * diff
    var = var / out_channels
    
    # Apply layer norm
    result_values = tl.zeros(out_channels, dtype=tl.float32)
    for oc in range(out_channels):
        gamma_val = tl.load(gamma_ptr + oc)
        beta_val = tl.load(beta_ptr + oc)
        
        norm_val = (conv_values[oc] - mean) / tl.sqrt(var + 1e-5)
        result_values[oc] = norm_val * gamma_val + beta_val
        
        # Store result
        tl.store(result_ptr + pos_offset + oc, result_values[oc])

@torch.fx.wrap
def fused_conv_norm(x, weight, bias, gamma, beta):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, kernel_size_2 = weight.shape
    
    assert kernel_size == 3 and kernel_size_2 == 3, "Only 3x3 kernels supported"
    assert in_height == 48 and in_width == 48, "Fixed input size expected"
    assert out_channels == 384, "Fixed output channels expected"
    
    # Calculate output dimensions  
    out_height = (in_height + 2*1 - 1*3 - 1) // 2 + 1  # 48 -> 24
    out_width = (in_width + 2*1 - 1*3 - 1) // 2 + 1   # 48 -> 24
    total_positions = out_height * out_width  # 24*24 = 576
    
    # Create output tensors
    # intermediate: [batch_size, out_channels, total_positions] = [1, 384, 576]
    intermediate = torch.zeros(batch_size, out_channels, total_positions, dtype=x.dtype, device=x.device)
    
    # result: [batch_size, total_positions, out_channels] = [1, 576, 384] 
    result = torch.zeros(batch_size, total_positions, out_channels, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    total_programs = batch_size * total_positions  # 1 * 576 = 576
    block_size = 64
    num_programs = (total_programs + block_size - 1) // block_size
    
    try:
        conv_norm_kernel[(num_programs,)](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            gamma_ptr=gamma,
            beta_ptr=beta,
            intermediate_ptr=intermediate,
            result_ptr=result,
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            in_height=in_height,
            in_width=in_width,
            BLOCK_HW=64
        )
    except Exception as e:
        print(f"Error launching Triton kernel: {e}")
        # If Triton kernel fails, let pass fail rather than use blocked APIs
        raise RuntimeError(f"Triton kernel failed for conv2d-layer_norm fusion: {e}")
    
    # Return intermediate (before layer norm) and result (after layer norm)
    # intermediate is [1, 384, 576] which matches the original permute shape
    # result is [1, 576, 384] which is the final layer norm output
    return intermediate, result

def replacement_func():
    return fused_conv_norm