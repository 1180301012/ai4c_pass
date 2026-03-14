import torch
import triton
import triton.language as tl

def pattern(tmp_0, tmp_1, in_2, in_3):
    # tmp_0 is bias, tmp_1 is weight, in_2 is input, in_3 is residual
    tmp_3 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    tmp_5 = in_3 + tmp_4
    return tmp_5, tmp_4

def replacement_args(tmp_0, tmp_1, in_2, in_3):
    return (tmp_0, tmp_1, in_2, in_3)

@triton.jit
def fused_linear_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    residual_ptr,
    out_ptr,
    dropout_out_ptr,
    n_batch,
    n_in_features,
    n_out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < n_batch
    n_mask = n_offsets < n_out_features
    
    bias = tl.load(bias_ptr)
    
    # Load input tensor [batch, in_features]
    input_offsets = (m_offsets[:, None] * n_in_features + tl.arange(0, BLOCK_SIZE_K)[None, :]).to(tl.int64)
    input_chunk = tl.load(input_ptr + input_offsets, mask=m_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < n_in_features), other=0.0)
    
    # Load weight tensor [out_features, in_features]
    weight_offsets = (n_offsets[None, :] * n_in_features + tl.arange(0, BLOCK_SIZE_K)[:, None]).to(tl.int64)
    weight_chunk = tl.load(weight_ptr + weight_offsets, mask=n_mask[None, :] & (tl.arange(0, BLOCK_SIZE_K)[:, None] < n_in_features), other=0.0)
    
    # Compute linear transformation: input @ weight.T + bias
    linear_result = tl.dot(input_chunk, weight_chunk, allow_tf32=False) + bias
    
    # Since dropout p=0.0, just copy the result
    dropout_result = linear_result
    
    # Load residual for addition
    residual_offsets = (m_offsets[:, None] * n_out_features + n_offsets[None, :]).to(tl.int64)
    residual_chunk = tl.load(residual_ptr + residual_offsets, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Add residual
    out_result = linear_result + residual_chunk
    
    # Store results
    out_offsets = (m_offsets[:, None] * n_out_features + n_offsets[None, :]).to(tl.int64)
    tl.store(out_ptr + out_offsets, out_result, mask=m_mask[:, None] & n_mask[None, :])
    
    # Store dropout result (no-op in this case since p=0.0)
    tl.store(dropout_out_ptr + out_offsets, dropout_result, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_linear_dropout_add(tmp_0, tmp_1, in_2, in_3):
    # Handle multi-dimensional tensors - apply linear on last dimension
    original_shape = in_2.shape
    n_in_features = in_2.shape[-1]
    n_out_features = tmp_1.shape[0]  # weight shape: [out_features, in_features]
    
    # Reshape input to 2D for matmul
    flattened_input = in_2.reshape(-1, n_in_features)
    flattened_residual = in_3.reshape(-1, n_out_features) if in_3.shape[-1] == n_out_features else in_3.reshape(-1)
    n_batch = flattened_input.shape[0]
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    num_m = (n_batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n = (n_out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensors
    out_flat = torch.empty(n_batch, n_out_features, dtype=in_2.dtype, device=in_2.device)
    dropout_out_flat = torch.empty(n_batch, n_out_features, dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    fused_linear_kernel[(num_m, num_n)](
        bias_ptr=tmp_0,
        weight_ptr=tmp_1,
        input_ptr=flattened_input,
        residual_ptr=flattened_residual,
        out_ptr=out_flat,
        dropout_out_ptr=dropout_out_flat,
        n_batch=n_batch,
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape back to original dimensions
    out_shape = (*original_shape[:-1], n_out_features)
    dropout_out_shape = out_shape
    
    return out_flat.reshape(out_shape), dropout_out_flat.reshape(dropout_out_shape)

def replacement_func():
    return fused_linear_dropout_add