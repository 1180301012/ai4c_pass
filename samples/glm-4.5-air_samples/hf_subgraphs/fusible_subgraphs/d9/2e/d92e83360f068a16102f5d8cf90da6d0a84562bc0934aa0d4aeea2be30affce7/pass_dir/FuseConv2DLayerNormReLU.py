import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    # The sequence: Conv2D → LayerNorm → ReLU
    # Match the exact computation from the model
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.conv2d(in_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_5 = torch.nn.functional.layer_norm(tmp_4, (tmp_4.shape[1], 1, 1), tmp_3, tmp_2, 1e-05)
    tmp_4 = tmp_3 = tmp_2 = None
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    tmp_5 = None
    # Return the same structure as the original model (tuple)
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # Extract all needed arguments for the fused kernel
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def fused_conv_norm_relu_kernel(
    input_ptr,           # Input tensor [N, C_in, 1, 1]
    weight_ptr,          # Conv weights [C_out, C_in, 1, 1] 
    bias_ptr,            # Conv bias [C_out]
    norm_weight_ptr,     # LayerNorm weight [C_out, 1, 1]
    norm_bias_ptr,       # LayerNorm bias [C_out, 1, 1]
    output_ptr,          # Output [N, C_out, 1, 1]
    N: tl.constexpr,      # Batch size
    C_in: tl.constexpr,   # Input channels
    C_out: tl.constexpr,  # Output channels
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Compute program ID
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Initialize offsets for batch and channel
    batch_start = pid_n * BLOCK_N
    channel_start = pid_c * BLOCK_C
    
    # Create offsets within the program
    batch_offset = batch_start + tl.arange(0, BLOCK_N)
    channel_offset = channel_start + tl.arange(0, BLOCK_C)
    
    # Create masks for bounds checking
    batch_mask = batch_offset < N
    channel_mask = channel_offset < C_out
    
    # If batch size is 1, we're working with single batch element
    if N == 1:
        batch_offset = 0
    
    # Load input tensor - shape [N, C_in, 1, 1]
    # For 1x1 conv with input [N, C_in, 1, 1], we're convolving over C_in dimension
    input_addr = input_ptr + (batch_offset[:, None] * C_in + channel_offset[None, :])[:, None]
    input_vals = tl.load(input_addr, mask=batch_mask[:, None], other=0.0)
    
    # Load convolution weights [C_out, C_in, 1, 1]
    weight_addr = weight_ptr + (channel_offset[:, None] * C_in + tl.arange(0, C_in)[None, :])[:, None]
    weight_vals = tl.load(weight_addr, mask=channel_mask[:, None], other=0.0)
    
    # Load convolution bias [C_out]
    conv_bias = tl.load(bias_ptr + channel_offset, mask=channel_mask, other=0.0)
    
    # Compute convolution: output = input @ weight^T + bias
    # For 1x1 conv on [N, C_in, 1, 1]: output[N, C_out] = sum_{C_in} input[N, C_in] * weight[C_out, C_in] + bias[C_out]
    conv_out = tl.sum(input_vals * weight_vals, axis=1) + conv_bias[None, :]
    
    # Load LayerNorm parameters
    norm_weight = tl.load(norm_weight_ptr + channel_offset, mask=channel_mask, other=1.0)
    norm_bias = tl.load(norm_bias_ptr + channel_offset, mask=channel_mask, other=0.0)
    
    # Apply LayerNorm: (x - mean) / std * weight + bias
    # For [N, C_out, 1, 1], we normalize over the C_out dimension per batch
    mean = tl.sum(conv_out, axis=1) / C_out
    variance = tl.sum((conv_out - mean[:, None]) ** 2, axis=1) / C_out
    std = tl.sqrt(variance + 1e-05)
    
    norm_out = (conv_out - mean[:, None]) / std[:, None] * norm_weight[None, :] + norm_bias[None, :]
    
    # Apply ReLU activation
    relu_out = tl.maximum(norm_out, 0.0)
    
    # Store output
    output_addr = output_ptr + (batch_offset[:, None] * C_out + channel_offset[None, :])[:, None]
    tl.store(output_addr, relu_out, mask=batch_mask[:, None])

@torch.fx.wrap
def fused_conv_norm_relu(in_0, in_1, in_2, in_3, in_4):
    # Get tensor shapes
    N, C_in, H_in, W_in = in_4.shape
    C_out = in_1.shape[0]  # Output channels from conv weight
    
    # Validate input shapes (should be 1x1 spatial dims for optimal performance)
    assert H_in == 1 and W_in == 1, f"Input spatial dimensions must be 1x1, got {H_in}x{W_in}"
    
    # Create output tensor
    Output = torch.empty((N, C_out, 1, 1), device=in_4.device, dtype=in_4.dtype)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_N = min(64, N) if N > 0 else 0
    BLOCK_C = min(128, C_out) if C_out > 0 else 0
    
    if N > 0 and C_out > 0:
        # Grid size: (num_batches, num_channels)
        grid = ( (N + BLOCK_N - 1) // BLOCK_N, (C_out + BLOCK_C - 1) // BLOCK_C )
        
        # Launch the fused kernel
        fused_conv_norm_relu_kernel[grid](
            in_4,      # input tensor
            in_1,      # conv weights
            in_0,      # conv bias
            in_3,      # norm weight
            in_2,      # norm bias
            Output,
            N,
            C_in,
            C_out,
            BLOCK_N,
            BLOCK_C
        )
    
    return Output

def replacement_func():
    return fused_conv_norm_relu