import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matches the entire computation graph:
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(in_2.shape[0], 64, 1, 1)  # Dynamic batch size
    tmp_5 = in_3 * tmp_4
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    batch_size = in_2.shape[0]
    tmp_4 = tmp_3.view(batch_size, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def linear_sigmoid_attention_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr,
    out_ptr,
    batch_size, feature_dim, height, width,
    weight_stride_0, weight_stride_1,
    bias_stride_0,
    in_3_stride_0, in_3_stride_1, in_3_stride_2, in_3_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Matrix multiplication part
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Bounds checking
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    m_mask = m_offsets < batch_size
    k_mask = k_offsets < feature_dim  # 8
    
    # Load input features [batch_size, 8]
    in_2_batch = tl.load(in_2_ptr + m_offsets[:, None] * 8 + k_offsets[None, :], 
                        mask=m_mask[:, None] & k_mask[None, :], 
                        other=0.0)
    
    # Load weights [64, 8] - transpose in memory for efficient matmul
    weights = tl.load(in_1_ptr + k_offsets[:, None] * 64 + tl.arange(0, 64)[None, :],
                      mask=k_mask[:, None] & (tl.arange(0, 64)[None, :] < 64),
                      other=0.0)
    weights_transposed = weights.T  # [8, 64]
    
    # Load bias [64]
    bias = tl.load(in_0_ptr + tl.arange(0, 64),
                   mask=tl.arange(0, 64) < 64,
                   other=0.0)
    
    # Matrix multiplication: [batch_size, 8] @ [8, 64] + [64]
    # Using blocked matrix multiplication for better performance
    linear_output = tl.zeros((BLOCK_SIZE_M, 64), dtype=tl.float32)
    
    # Split K dimension into blocks
    K = 8  # feature_dim
    for k in range(0, K, BLOCK_SIZE_K):
        # Load weight block
        k_end = min(k + BLOCK_SIZE_K, K)
        k_mask_block = (k + tl.arange(0, min(BLOCK_SIZE_K, K - k))) < K
        
        # Load weight block [BLOCK_SIZE_K, 64]
        weight_block = tl.load(in_1_ptr + 
                             (k + tl.arange(0, min(BLOCK_SIZE_K, K - k)))[:, None] * 64 +  
                             tl.arange(0, 64)[None, :],
                             mask=k_mask_block[:, None] & (tl.arange(0, 64)[None, :] < 64),
                             other=0.0)
        weight_block_transposed = weight_block.T  # [64, BLOCK_SIZE_K]
        
        # Load input block [BLOCK_SIZE_M, BLOCK_SIZE_K]
        in_2_block = tl.load(in_2_ptr + 
                            m_offsets[:, None] * K + 
                            (k + tl.arange(0, min(BLOCK_SIZE_K, K - k)))[None, :],
                            mask=m_mask[:, None] & k_mask_block[None, :],
                            other=0.0)
        
        # Accumulate matmul result
        linear_output += tl.dot(in_2_block, weight_block_transposed)
    
    # Add bias
    linear_output += bias[None, :]
    
    # Apply sigmoid activation
    sigmoid_out = 1.0 / (1.0 + tl.exp(-linear_output))
    
    # Broadcast across spatial dimensions and multiply with input feature map
    h_offsets = tl.program_id(2) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = tl.program_id(3) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_mask = h_offsets < height
    w_mask = w_offsets < width
    
    # Broadcast sigmoid result to [batch_size, 64, height, width]
    sigmoid_expanded = sigmoid_out[:, :, None, None]
    
    # Load feature map [batch_size, 64, height, width]
    in_3_batch = tl.load(in_3_ptr + 
                        m_offsets[:, None, None, None] * in_3_stride_0 +
                        64 * tl.arange(0, 64)[None, None, None, :1] * in_3_stride_1 +
                        h_offsets[None, None, :, None] * in_3_stride_2 +
                        w_offsets[None, None, None, :] * in_3_stride_3,
                        mask=(m_mask[:, None, None, None] & 
                             (tl.arange(0, 64)[None, None, None, :1] < 64) &
                             h_mask[None, None, :, None] &
                             w_mask[None, None, None, :]),
                        other=0.0)
    
    # Element-wise multiplication
    out = in_3_batch * sigmoid_expanded
    
    # Store result
    tl.store(out_ptr + 
             m_offsets[:, None, None, None] * out_stride_0 +
             64 * tl.arange(0, 64)[None, None, None, :1] * out_stride_1 +
             h_offsets[None, None, :, None] * out_stride_2 +
             w_offsets[None, None, None, :] * out_stride_3,
             out,
             mask=(m_mask[:, None, None, None] & 
                  (tl.arange(0, 64)[None, None, None, :1] < 64) &
                  h_mask[None, None, :, None] &
                  w_mask[None, None, None, :]))

@torch.fx.wrap
def linear_sigmoid_attention_fused(in_0, in_1, in_2, in_3):
    batch_size, feature_dim = in_2.shape
    _, channels, height, width = in_3.shape
    
    # Output tensor
    out = torch.empty_like(in_3)
    
    # Autotune block sizes based on input dimensions for optimal performance
    if batch_size <= 4:
        BLOCK_SIZE_M = 4
    elif batch_size <= 32:
        BLOCK_SIZE_M = 8
    else:
        BLOCK_SIZE_M = 16
    
    BLOCK_SIZE_N = 64  # Output channels
    BLOCK_SIZE_K = 8   # Input features feature_dim
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    
    # Calculate grid size
    num_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_k = (feature_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    num_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_w = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    grid = (num_m, num_k, num_h, num_w)
    
    linear_sigmoid_attention_kernel[grid](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        in_3_ptr=in_3,
        out_ptr=out,
        batch_size=batch_size,
        feature_dim=feature_dim,
        height=height,
        width=width,
        weight_stride_0=64,
        weight_stride_1=8,
        bias_stride_0=1,
        in_3_stride_0=channels * height * width,
        in_3_stride_1=height * width,
        in_3_stride_2=width,
        in_3_stride_3=1,
        out_stride_0=channels * height * width,
        out_stride_1=height * width,
        out_stride_2=width,
        out_stride_3=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return linear_sigmoid_attention_fused