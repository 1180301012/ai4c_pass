import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_slice_reshape_kernel_1(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out1_ptr,
    out2_ptr,
    M,
    N,  # Should be 512 (input features)
    K,  # Should be 256 (output features)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = M // BLOCK_SIZE_M
    grid_n = K // BLOCK_SIZE_N
    
    # Process first half (first 256 output features)
    mid_pid = tl.program_id(1)
    if mid_pid == 0:
        block_m = pid
        block_n = tl.arange(0, BLOCK_SIZE_N)
        block_m_offsets = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        block_n_offsets = block_n * BLOCK_SIZE_N
        
        mask = (block_m_offsets[:, None] < M) & (block_n_offsets[None, :] < K)
        
        # Load input
        input_ptrs = input_ptr + block_m_offsets[:, None] * N + block_n_offsets[None, :]
        input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
        
        # Load weight
        weight_ptrs = weight_ptr + block_n_offsets[:, None] * N
        weight_vals = tl.load(weight_ptrs, mask=(block_n_offsets[:, None] < K) & (tl.arange(0, N)[None, :] < N), other=0.0)
        
        # Linear transformation
        output = tl.dot(input_vals.to(tl.float32), weight_vals.to(tl.float32))
        
        # Add bias
        bias_ptrs = bias_ptr + block_n_offsets
        bias_vals = tl.load(bias_ptrs, mask=block_n_offsets < K, other=0.0)
        output = output + bias_vals[None, :]
        
        # Store first half output (first 256 features)
        output_ptrs = out1_ptr + block_m_offsets[:, None] * 256 + block_n_offsets[None, :]
        tl.store(output_ptrs, output, mask=mask)

@triton.jit
def fused_linear_slice_reshape_kernel_2(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out1_ptr,
    out2_ptr,
    M,
    N,  # Should be 512 (input features)
    K,  # Should be 512 (output features)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = M // BLOCK_SIZE_M
    grid_n = K // BLOCK_SIZE_N
    
    block_m = pid
    block_n = tl.arange(0, BLOCK_SIZE_N)
    block_m_offsets = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_n_offsets = block_n * BLOCK_SIZE_N
    
    mask = (block_m_offsets[:, None] < M) & (block_n_offsets[None, :] < K)
    
    # Load input
    input_ptrs = input_ptr + block_m_offsets[:, None] * N + block_n_offsets[None, :]
    input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Load weight split into two blocks
    # First block (first 256 features)
    weight1_ptrs = weight_ptr + (block_n_offsets % 256)[:, None] * N + (block_n_offsets // 256)[:, None] * 256
    weight1_vals = tl.load(weight1_ptrs, mask=(block_n_offsets % 256 < 256) & (tl.arange(0, N)[None, :] < N), other=0.0)
    
    # Second block (last 256 features)
    weight2_ptrs = weight_ptr + ((block_n_offsets % 256) + 256)[:, None] * N + (block_n_offsets // 256)[:, None] * 256
    weight2_vals = tl.load(weight2_ptrs, mask=(block_n_offsets % 256 < 256) & (tl.arange(0, N)[None, :] < N), other=0.0)
    
    # Linear transformations
    output1 = tl.dot(input_vals.to(tl.float32), weight1_vals.to(tl.float32))
    bias1 = tl.load(bias_ptr + (block_n_offsets % 256), mask=(block_n_offsets % 256 < 256), other=0.0)
    output1 = output1 + bias1[None, :]
    
    output2 = tl.dot(input_vals.to(tl.float32), weight2_vals.to(tl.float32))
    bias2 = tl.load(bias_ptr + ((block_n_offsets % 256) + 256), mask=(block_n_offsets % 256 < 256), other=0.0)
    output2 = output2 + bias2[None, :]
    
    # Store outputs
    feature_indices = block_n_offsets % 256
    out1_ptrs = out1_ptr + block_m_offsets[:, None] * 256 + feature_indices[None, :]
    out2_ptrs = out2_ptr + block_m_offsets[:, None] * 256 + feature_indices[None, :]
    
    tl.store(out1_ptrs, output1, mask=(block_m_offsets[:, None] < M) & (feature_indices[None, :] < 256))
    tl.store(out2_ptrs, output2, mask=(block_m_offsets[:, None] < M) & (feature_indices[None, :] < 256))

@torch.fx.wrap
def fused_linear_slice_reshape_1(in_5, in_1, in_0):
    M, N = in_5.shape  # Should be 300, 256
    weight_m, weight_n = in_1.shape  # Should be 512, 256
    
    # Output shapes
    out1_shape = (M * (N // 256), 256) if N >= 256 else (M, 256)
    out2_shape = (M * (N // 256), 256) if N >= 256 else (M, 256)
    
    out1 = torch.empty(out1_shape, dtype=in_5.dtype, device=in_5.device)
    out2 = torch.empty(out2_shape, dtype=in_5.dtype, device=in_5.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (256 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_linear_slice_reshape_kernel_1[(grid_m, 1)](
        in_5, in_1, in_0, out1, out2, M, N, 256,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out1, out2

@torch.fx.wrap
def fused_linear_slice_reshape_2(tmp_9, in_3, in_2):
    # Get batch dimension and reshape to matrix for linear operation
    original_shape = tmp_9.shape
    M = tmp_9.shape[0]  # 300
    N = tmp_9.shape[1] * tmp_9.shape[2] if len(tmp_9.shape) > 2 else tmp_9.shape[1]
    actual_features = 512  # From weight shape
    
    weight_m, weight_n = in_3.shape  # Should be 512, 256
    
    # Reshape input to matrix
    input_matrix = tmp_9.reshape(-1, N) if len(tmp_9.shape) > 2 else tmp_9
    
    # Output shapes
    out1_shape = original_shape[:-1] + (256,)
    out2_shape = original_shape[:-1] + (256,)
    
    out1 = torch.empty(out1_shape, dtype=tmp_9.dtype, device=tmp_9.device)
    out2 = torch.empty(out2_shape, dtype=tmp_9.dtype, device=tmp_9.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    M_total = input_matrix.shape[0]
    grid_m = (M_total + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (512 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_linear_slice_reshape_kernel_2[(grid_m, 2)](
        input_matrix, in_3, in_2, out1, out2, M_total, N, 512,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out1, out2

def pattern(in_5, in_1, in_0):
    tmp_4 = in_5 @ in_1 + in_0  # Linear operation: X @ W^T + b
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    return tmp_6, tmp_8

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

def replacement_func():
    return fused_linear_slice_reshape_1