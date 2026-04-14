import torch
import triton
import triton.language as tl



def pattern(in_0, in_1, in_2):
    # Match both patterns: some models use tmp_0=in_0, tmp_1=in_1, others use direct args
    # Pattern 1: Some models have intermediate assignments
    tmp_0 = in_0
    tmp_1 = in_1
    linear_result = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    
    # Permute and reshape
    permuted = linear_result.permute(0, 2, 1)
    batch_size = in_2.shape[0]
    reshaped = permuted.reshape(batch_size, -1, 16, 16)
    
    return reshaped

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_linear_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles (BLOCK_M, BLOCK_N) elements of the output tensor
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)  # height dimension (16)
    pid_w = tl.program_id(3)  # width dimension (16)
    
    # Compute base offsets in the flattened output
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Only process elements within bounds
    m_mask = m_offsets < batch_size
    n_mask = n_offsets < out_features
    
    # Load bias for this output feature
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Compute the spatial position index (0 to 255 for 16x16)
    spatial_pos = pid_h * 16 + pid_w
    
    if spatial_pos >= seq_len:
        # If this spatial position is beyond the sequence length, output zeros
        for i in range(BLOCK_M):
            for j in range(BLOCK_N):
                if m_offsets[i] < batch_size and n_offsets[j] < out_features:
                    output_idx = (m_offsets[i] * out_features * 16 * 16 + 
                                 n_offsets[j] * 16 * 16 + 
                                 pid_h * 16 + pid_w)
                    tl.store(output_ptr + output_idx, 0.0)
        return
    
    # Load the corresponding input token
    input_token = tl.load(
        input_ptr + m_offsets[:, None] * seq_len * in_features + 
        spatial_pos * in_features + tl.arange(0, in_features)[None, :],
        mask=(m_offsets[:, None] < batch_size)[:, None],
        other=0.0
    )
    
    # Load corresponding weights
    weight_token = tl.load(
        weight_ptr + n_offsets[:, None] * in_features + tl.arange(0, in_features)[None, :],
        mask=(n_offsets[:, None] < out_features)[:, None],
        other=0.0
    )
    
    # Compute dot product
    result = tl.sum(input_token * weight_token, axis=1)
    
    # Add bias and store
    result = result + bias
    
    # Store output
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if m_offsets[i] < batch_size and n_offsets[j] < out_features:
                output_idx = (m_offsets[i] * out_features * 16 * 16 + 
                             n_offsets[j] * 16 * 16 + 
                             pid_h * 16 + pid_w)
                tl.store(output_ptr + output_idx, result[i * BLOCK_N + j])

@torch.fx.wrap
def fused_linear_permute_reshape(bias, weight, input_tensor):
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    in_features = input_tensor.shape[2]
    out_features = bias.shape[0]
    
    # Output shape: [batch_size, out_features, 16, 16]
    output_shape = (batch_size, out_features, 16, 16)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions
    BLOCK_M = 8   # Process 8 batch elements per program
    BLOCK_N = 64  # Process 64 output features per program
    BLOCK_K = in_features  # Process all input features at once
    
    grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_features + BLOCK_N - 1) // BLOCK_N
    grid_h = 16  # Fixed height dimension
    grid_w = 16  # Fixed width dimension
    grid = (grid_m, grid_n, grid_h, grid_w)
    
    fused_linear_kernel[grid](
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return output

def replacement_func():
    return fused_linear_permute_reshape