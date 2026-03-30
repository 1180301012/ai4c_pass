import torch
import triton
import triton.language as tl

def pattern(conv_input, weight_tensor, bias_tensor):
    """
    Pattern matching for the attention mechanism: Conv2D → View → Softmax → Unsqueeze
    This pattern is common in attention mechanisms where:
    - Conv2D applies attention projections (typically 1x1 conv)
    - View reshapes for softmax application
    - Softmax normalizes attention weights
    - Unsqueeze adds dimension for subsequent operations
    """
    conv2d = torch.conv2d(conv_input, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(conv2d.shape[0], conv2d.shape[1], -1)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return tmp_5

def replacement_args(conv_input, weight_tensor, bias_tensor):
    return (conv_input, weight_tensor, bias_tensor)

@triton.jit
def attention_conv_softmax_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, num_channels, height, width,
    weight_channels_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized Triton kernel for fused attention mechanism:
    Conv2D (1x1) → View → Softmax → Unsqueeze
    
    This kernel fuses multiple operations to reduce memory bandwidth usage
    and improve computational efficiency.
    """
    # Matrix dimensions for matrix multiplication
    M = batch_size * height * width  # Total spatial positions
    N = weight_channels_out  # Output channels
    K = num_channels  # Input channels
    
    # Program ID for matrix multiplication
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    row_offset = pid // grid_n
    col_offset = pid % grid_n
    
    # compute range each program should process
    row_start = row_offset * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_start = col_offset * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = row_start < M
    col_mask = col_start < N
    
    # re-order operations to reduce memory accesses
    if row_mask[0] and col_mask[0]:
        # Load input data (spatial positions x input channels)
        input_offset = row_start[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
        input_mask = (row_start[:, None] < M)[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < K)
        x = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
        
        # Load weight data (input channels x output channels)
        weight_offset = tl.arange(0, BLOCK_SIZE_K)[:, None] * N + col_start[None, :]
        weight_mask = (tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & (col_start[None, :] < N)
        w = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32)
        
        # Load bias data
        bias_offset = col_start[None, :]
        bias_mask = (col_start[None, :] < N)
        b = tl.load(bias_ptr + bias_offset, mask=bias_mask, other=0.0).to(tl.float32)
        
        # Matrix multiplication: conv2d equivalent
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            # Load a block of the input matrix
            x_block = x[:, k:k + BLOCK_SIZE_K]
            w_block = w[k:k + BLOCK_SIZE_K, :]
            
            # accumulate results
            acc += tl.dot(x_block, w_block)
        
        # Add bias
        acc = acc + b[None, :]
        
        # Apply softmax over the last dimension (which corresponds to flattened spatial features)
        # Reshape to apply softmax spatially
        acc_reshaped = acc.reshape(batch_size, height * width, N)
        
        # Compute softmax along the flattened spatial dimension
        max_val = tl.max(acc_reshaped, axis=1, keepdims=True)
        exp_val = tl.exp(acc_reshaped - max_val)
        sum_val = tl.sum(exp_val, axis=1, keepdims=True)
        softmax_result = exp_val / (sum_val + 1e-10)  # Add small epsilon for numerical stability
        
        # Reshape back and add unsqueeze dimension
        softmax_result = softmax_result.reshape(batch_size, N, height, width)
        softmax_result = softmax_result.unsqueeze(-1)
        
        # Store output
        output_offset = row_start[:, None] * N * 1 + col_start[None, :] * 1
        output_mask = (row_start[:, None] < M) & (col_start[None, :] < N)
        tl.store(output_ptr + output_offset, softmax_result, mask=output_mask)

@torch.fx.wrap
def attention_conv_softmax_fused(conv_input, weight_tensor, bias_tensor):
    """
    Fused attention mechanism implementation combining:
    1x1 Conv2D → View → Softmax → Unsqueeze
    
    This implementation reduces memory bandwidth usage by fusing multiple
    operations and eliminating intermediate memory allocations.
    """
    input_shape = conv_input.shape
    batch_size, num_channels, height, width = input_shape
    weight_channels_out = weight_tensor.shape[0]
    
    output_shape = (batch_size, weight_channels_out, height, width, 1)
    output = torch.empty(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Optimized block sizes for GPU architecture
    BLOCK_SIZE_M = 64  # Spatial positions per block
    BLOCK_SIZE_N = 64  # Output channels per block  
    BLOCK_SIZE_K = 32  # Input channels per block (vectorizes the computation)
    
    num_programs = ((batch_size * height * width + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * \
                   ((weight_channels_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    if num_programs > 0:
        attention_conv_softmax_kernel[(num_programs,)](
            input_ptr=conv_input,
            weight_ptr=weight_tensor,
            bias_ptr=bias_tensor,
            output_ptr=output,
            batch_size=batch_size,
            num_channels=num_channels,
            height=height,
            width=width,
            weight_channels_out=weight_channels_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    
    return output

def replacement_func():
    return attention_conv_softmax_fused