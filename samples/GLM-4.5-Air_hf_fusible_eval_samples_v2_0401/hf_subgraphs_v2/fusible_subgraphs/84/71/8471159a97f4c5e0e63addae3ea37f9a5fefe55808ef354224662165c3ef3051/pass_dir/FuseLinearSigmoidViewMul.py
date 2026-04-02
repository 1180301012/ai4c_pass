import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Exact pattern matching: Linear + Sigmoid + View + Element-wise Multiplication"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    # Based on the model pattern, the view shape should be [in_2.shape[0], 64, 1, 1]
    # This handles all cases: 1, 32, or 128 batch sizes
    tmp_4 = tmp_3.view(in_2.shape[0], 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_sigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs for matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute M and N indices
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input features)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load x slice: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        x_mask = x_offsets < M
        k_mask = k_offsets < K
        
        x_values = tl.load(x_ptr + x_offsets[:, None] * K + k_offsets[None, :], 
                          mask=x_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load weight slice: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        w_offsets = k_offsets[:, None] * N + n_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]
        w_mask = k_offsets[:, None] & k_mask[:, None] & (n_offset + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
        w_values = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0)
        
        # Matrix multiply: accumulate outer product
        accumulator += tl.dot(x_values, w_values)
    
    # Load bias for this output column
    bias_values = tl.load(bias_ptr + n_offset + tl.arange(0, BLOCK_SIZE_N), 
                         mask=n_offset + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
    
    # Add bias and apply sigmoid
    accumulator = accumulator + bias_values[None, :]
    accumulator = 1.0 / (1.0 + tl.exp(-accumulator))
    
    # Store result
    out_offsets = (m_offset + tl.arange(0, BLOCK_SIZE_M))[:, None] * N + (n_offset + tl.arange(0, BLOCK_SIZE_N))[None, :]
    out_mask = (m_offset + tl.arange(0, BLOCK_SIZE_M))[:, None] & (n_offset + tl.arange(0, BLOCK_SIZE_N))[None, :] < M
    tl.store(out_ptr + out_offsets, accumulator, mask=out_mask)

@triton.jit
def elementwise_mul_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    shape_batch, shape_channels, shape_height, shape_width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < (shape_batch * shape_channels * shape_height * shape_width)
    
    # Load values from linear+sigmoid operation
    x_val = tl.load(x_ptr + offsets, mask=mask)
    
    # Load y values - need to broadcast to spatial shape
    # For spatial tensors, we need special handling to match the view operation
    if shape_height > 1 or shape_width > 1:
        # Broadcast the linear+sigmoid output which has spatial shape [B, C, 1, 1] to [B, C, H, W]
        # The multiplication pattern is: spatials * sigmoid_out_reshaped
        # where sigmoid_out_reshaped has shape [B, C, 1, 1] and is broadcast
        
        # Calculate the indices for broadcasting
        batch_idx = offsets // (shape_channels * shape_height * shape_width)
        channel_idx = (offsets // (shape_height * shape_width)) % shape_channels
        h_idx = (offsets // shape_width) % shape_height
        w_idx = offsets % shape_width
        
        # For the sigmoid output (which is [B, C]):
        sigmoid_offset = batch_idx * shape_channels + channel_idx
        sigmoid_ptr = x_ptr + sigmoid_offset * (shape_height * shape_width)
        sigmoid_val = tl.load(sigmoid_ptr, mask=sigmoid_offset < (shape_batch * shape_channels), other=0.0)
        
        # For y tensor, just load directly
        y_val = tl.load(y_ptr + offsets, mask=mask)
        
        # The multiplication is: y * sigmoid_broadcasted
        # Since sigmoid is [B, C] and we need [B, C, 1, 1], it gets broadcast
        # The element we want is: y[b,c,h,w] * sigmoid[b,c]
        out = y_val * sigmoid_val
    else:
        # Simple case - no spatial broadcasting needed
        y_val = tl.load(y_ptr + offsets, mask=mask)
        out = x_val * y_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def fused_linear_sigmoid_view_mul(in_0, in_1, in_2, in_3):
    """Fused kernel for Linear + Sigmoid + View + Element-wise Multiplication"""
    
    # Get tensor shapes
    bias = in_0  # [64]
    weight = in_1  # [64, 8]
    x = in_2  # input with shape [M, 8] or variants
    y = in_3  # spatial tensor with shape [B, C, H, W]
    
    M = x.shape[0]  # batch size from input
    N = weight.shape[0]  # output features (64)
    K = weight.shape[1]  # input features (8)
    
    # Determine spatial dimensions from y tensor
    if len(y.shape) == 4:
        batch_size_orig, channels_orig, height_orig, width_orig = y.shape
    else:
        raise ValueError("Unsupported y shape")
    
    # Step 1: Linear transformation + sigmoid
    # Output shape will be [M, 64]
    out_linear_sigmoid = torch.empty(M, N, dtype=x.dtype, device=x.device)
    
    # Tile sizes for matrix multiplication
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 8
    
    # Handle cases where batch size doesn't divide evenly by block size
    if M % BLOCK_SIZE_M != 0:
        M_eff = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * BLOCK_SIZE_M
    else:
        M_eff = M
    
    # Launch kernel for matrix multiplication + sigmoid
    grid_m = (M_eff + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_linear_sigmoid_kernel[(grid_m, grid_n)](
        x.data_ptr(),
        weight.data_ptr(), 
        bias.data_ptr(),
        out_linear_sigmoid.data_ptr(),
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # If we padded, slice to original size
    if M_eff > M:
        out_linear_sigmoid = out_linear_sigmoid[:M, :]
    
    # Step 2: Reshape sigmoid output to spatial format
    # Based on the model patterns, this should be [M, 64, 1, 1]
    sigmoid_spatial = out_linear_sigmoid.view(M, N, 1, 1)
    
    # Step 3: Broadcast to match spatial dimensions and multiply
    # The operation is: y * sigmoid_spatial (which gets broadcast automatically)
    out_final = y * sigmoid_spatial
    
    return out_final

def replacement_func():
    return fused_linear_sigmoid_view_mul