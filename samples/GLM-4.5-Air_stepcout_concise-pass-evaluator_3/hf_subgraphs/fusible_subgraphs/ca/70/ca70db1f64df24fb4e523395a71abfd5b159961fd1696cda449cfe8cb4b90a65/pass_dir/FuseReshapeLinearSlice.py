import torch
import triton
import triton.language as tl

def pattern_reshape_linear_slice(input_tensor, weight_tensor, bias_tensor):
    tmp_9 = input_tensor.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, weight_tensor, bias_tensor)
    tmp_11 = tmp_10[..., slice(None, 256, None)]
    tmp_12 = tmp_10[..., slice(-256, None, None)]
    return tmp_11, tmp_12

@triton.jit
def fused_reshape_linear_slice_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out1_ptr,
    out2_ptr,
    batch_size,
    seq_len,
    input_features,
    output_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = batch_size * seq_len
    grid_n = output_features
    
    block_m = pid
    block_n = tl.arange(0, BLOCK_SIZE_N)
    block_m_offsets = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_n_offsets = block_n * BLOCK_SIZE_N
    
    # Calculate element positions
    batch_offsets = (block_m_offsets // seq_len) % batch_size
    seq_offsets = block_m_offsets % seq_len
    
    # Calculate input pointer offsets
    input_offsets = batch_offsets[:, None] * (seq_len * input_features) + seq_offsets[:, None] * input_features + (block_n_offsets % 256)[None, :]
    
    # Calculate mask
    mask = (block_m_offsets < grid_m) & (block_n_offsets < 256)  # We only need first 256 features
    
    # Load input
    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Load weight split into first 256 and last 256 features
    weight1_ptrs = weight_ptr + (block_n_offsets % 256)[:, None] * input_features
    weight1_vals = tl.load(weight1_ptrs, mask=(block_n_offsets % 256 < 256) & (tl.arange(0, input_features)[None, :] < input_features), other=0.0)
    
    weight2_ptrs = weight_ptr + ((block_n_offsets % 256) + 256)[:, None] * input_features
    weight2_vals = tl.load(weight2_ptrs, mask=(block_n_offsets % 256 < 256) & (tl.arange(0, input_features)[None, :] < input_features), other=0.0)
    
    # Linear transformations for both slices
    output1 = tl.dot(input_vals.to(tl.float32), weight1_vals.to(tl.float32))
    bias1 = tl.load(bias_ptr + (block_n_offsets % 256), mask=(block_n_offsets % 256 < 256), other=0.0)
    output1 = output1 + bias1[None, :]
    
    output2 = tl.dot(input_vals.to(tl.float32), weight2_vals.to(tl.float32))
    bias2 = tl.load(bias_ptr + ((block_n_offsets % 256) + 256), mask=(block_n_offsets % 256 < 256), other=0.0)
    output2 = output2 + bias2[None, :]
    
    # Reshape and store outputs
    out1_shape = (batch_size, seq_len, 256)
    out2_shape = (batch_size, seq_len, 256)
    
    # Convert to output coordinates
    out1_batch = (block_m_offsets // seq_len) % batch_size
    out1_seq = block_m_offsets % seq_len
    out1_feat = block_n_offsets % 256
    
    out2_batch = (block_m_offsets // seq_len) % batch_size
    out2_seq = block_m_offsets % seq_len
    out2_feat = block_n_offsets % 256
    
    # Store outputs
    out1_ptrs = out1_ptr + out1_batch[:, None] * (seq_len * 256) + out1_seq[:, None] * 256 + out1_feat[None, :]
    out2_ptrs = out2_ptr + out2_batch[:, None] * (seq_len * 256) + out2_seq[:, None] * 256 + out2_feat[None, :]
    
    mask_out = mask & (block_n_offsets < 512)  # Ensure we're within valid output features
    
    tl.store(out1_ptrs, output1, mask=(out1_batch[:, None] < batch_size) & (out1_seq[:, None] < seq_len) & (out1_feat[None, :] < 256))
    tl.store(out2_ptrs, output2, mask=(out2_batch[:, None] < batch_size) & (out2_seq[:, None] < seq_len) & (out2_feat[None, :] < 256))

@torch.fx.wrap
def fused_reshape_linear_slice(in_4, in_3, in_2):
    # Input shape: [1, 150, 1, 512] -> reshape to [300, -1, 256]
    input_shape = in_4.shape  # [1, 150, 1, 512]
    batch_size = input_shape[0] * input_shape[1] * input_shape[2]  # 1 * 150 * 1 = 300
    seq_len = input_shape[3] // 256 if len(input_shape) > 3 else 1  # 512 / 256 = 2
    input_features = 256  # Per feature dimension
        
    # Weight shape: [512, 256] -> [2, 256, 256] after splitting
    weight_m, weight_n = in_3.shape  # Should be 512, 256
    
    # Output shapes
    out1_shape = (batch_size, seq_len, 256)  # tmp_11
    out2_shape = (batch_size, seq_len, 256)  # tmp_12
    
    out1 = torch.empty(out1_shape, dtype=in_4.dtype, device=in_4.device)
    out2 = torch.empty(out2_shape, dtype=in_4.dtype, device=in_4.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    total_elements = batch_size * seq_len
    grid_size = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    fused_reshape_linear_slice_kernel[grid_size, 2](
        in_4, in_3, in_2, out1, out2,
        batch_size, seq_len, input_features, output_features=512,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out1, out2

def pattern(in_4, in_3, in_2):
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = tmp_9 @ in_3 + in_2  # Linear operation: X @ W^T + b
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    return tmp_11, tmp_12

def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)

def replacement_func():
    return fused_reshape_linear_slice