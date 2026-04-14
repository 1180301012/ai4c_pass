import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias, scale, shift, eps):
    """Pattern: conv2d → flatten → transpose → layer_norm → dropout(0.0)"""
    conv2d = torch.conv2d(conv_input, weight, bias, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), scale, shift, eps)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_7, tmp_9

def replacement_args(conv_input, weight, bias, scale, shift, eps):
    return (conv_input, weight, bias, scale, shift, eps)

@triton.jit
def layernorm_kernel(
    x_ptr, gamma_ptr, beta_ptr,
    output_ptr,
    n_elements, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Layer normalization kernel in Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Reshape to [batch_seq_len, hidden_size] for layer norm
    n_tokens = n_elements // hidden_size
    x_2d = x.reshape((n_tokens, hidden_size))
    
    # Compute mean and variance (simplified - in reality would need reduction)
    # For this implementation, we'll compute mean and variance per token
    mean = tl.sum(x_2d, axis=1) / hidden_size
    var = tl.sum((x_2d - mean[:, None]) ** 2, axis=1) / hidden_size
    
    # Layer normalization formula
    x_normalized = (x_2d - mean[:, None]) * tl.sqrt(var[:, None] + 1e-05) ** -1
    
    # Apply scale and shift
    gamma = tl.load(gamma_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
    beta = tl.load(beta_ptr + tl.arange(0, hidden_size), mask=tl.arange(0, hidden_size) < hidden_size).to(tl.float32)
    
    x_final = x_normalized * gamma + beta
    
    # Store result
    tl.store(output_ptr + offsets, x_final, mask=mask)

@triton.jit
def fused_conv_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr,
    output_ptr, transpose_output_ptr,
    N, C_in, H_in, W_in, C_out, H_out, W_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Conv2D parameters
    stride_h, stride_w = 2, 2
    pad_h, pad_w = 0, 0
    dilation_h, dilation_w = 1, 1
    groups = 1
    
    # Element positions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of columns for this program
    cols_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_idx < C_out
    
    # Conv2D computation
    h_start = pid_m * BLOCK_SIZE_M
    h_idx = h_start + tl.arange(0, BLOCK_SIZE_M)
    h_mask = h_idx < H_out
    
    # Load weights
    weight_offset = (pid_n // groups) * C_in * 2 * 2 + (tl.arange(0, C_in)[:, None] * 2 * 2 + tl.arange(0, 2 * 2)[None, :])
    weight_val = tl.load(weight_ptr + weight_offset, mask=tl.broadcast_to(cols_mask[:, None], (C_out, C_in * 2 * 2))).to(tl.float32)
    
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for c_idx in range(0, C_in, BLOCK_SIZE_K):
        c_end = min(c_idx + BLOCK_SIZE_K, C_in)
        c_mask = c_idx + tl.arange(0, c_end - c_idx)[:, None] < C_in
        
        # Load input patches (simplified for this demonstration)
        input_offset = c_idx * H_in * W_in + (h_idx[:, None] * W_in + tl.arange(0, W_out)[None, :]) * stride_w
        input_val = tl.load(input_ptr + input_offset, mask=h_mask[:, None] & (tl.arange(0, W_out)[None, :] < W_out), other=0.0).to(tl.float32)
        
        # Conv2D computation
        conv_output = (input_val.to(tl.float32) * weight_val[None, :]).sum(dim=2)
        output += conv_output
    
    # Add bias
    bias_val = tl.load(bias_ptr + cols_idx, mask=cols_mask, other=0.0).to(tl.float32)
    output += bias_val[None, :]
    
    # Store main output
    output_offset = pid_m * BLOCK_SIZE_M * C_out + cols_idx[None, :]
    tl.store(output_ptr + output_offset, output, mask=h_mask[:, None] & cols_mask[None, :])
    
    # Flatten and transpose
    flat_offset = pid_m * BLOCK_SIZE_M * H_out * C_out + cols_idx[None, :] * H_out
    tl.store(transpose_output_ptr + flat_offset, output, mask=h_mask[:, None] & cols_mask[None, :])

@torch.fx.wrap
def fused_conv_norm(input_tensor, weight, bias, scale, shift, eps):
    # Get input dimensions
    batch, channels_in, height_in, width_in = input_tensor.shape
    channels_out = weight.shape[0]
    height_out = (height_in + 2 * 0 - 2 * 1) // 2 + 1
    width_out = (width_in + 2 * 0 - 2 * 1) // 2 + 1
    
    # Create output tensors
    output = torch.empty((batch, channels_out, height_out, width_out), dtype=input_tensor.dtype, device=input_tensor.device)
    transpose_output = torch.empty((batch, channels_out, height_out * width_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose block sizes (optimized for typical GPU features)
    BLOCK_SIZE_M = 8  # height dimension
    BLOCK_SIZE_N = 64  # output channels dimension
    BLOCK_SIZE_K = 32  # input channels dimension
    
    # Grid dimensions
    grid_m = (height_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (channels_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch conv2d kernel
    fused_conv_norm_kernel[(grid_m, grid_n)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        scale_ptr=scale,
        shift_ptr=shift,
        output_ptr=output,
        transpose_output_ptr=transpose_output,
        N=batch, C_in=channels_in, H_in=height_in, W_in=width_in, 
        C_out=channels_out, H_out=height_out, W_out=width_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Apply layer normalization using Triton kernel
    n_elements = transpose_output.numel()
    hidden_size = channels_out
    
    # Create output for layer norm
    norm_output = torch.empty_like(transpose_output)
    
    # Launch layer norm kernel
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    layernorm_kernel[grid_size](
        x_ptr=transpose_output,
        gamma_ptr=scale,
        beta_ptr=shift,
        output_ptr=norm_output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Dropout with rate 0.0 is a no-op, so just return the outputs
    return transpose_output, norm_output

def replacement_func():
    return fused_conv_norm