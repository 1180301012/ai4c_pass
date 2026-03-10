import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Basic pattern - linear + element-wise multiplication
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = in_2 * tmp_1
    return (tmp_2,)



def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def linear_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    n_batch,
    n_seq,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    pid_seq = tl.program_id(3)
    
    # Calculate ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create ranges for matrices
    ranges_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    ranges_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    ranges_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Broadcast ranges for proper indexing
    ranges_m = ranges_m[:, None]
    ranges_n = ranges_n[None, :]
    ranges_k = ranges_k[None, None, :]
    
    mask_m = ranges_m < n_rows
    mask_n = ranges_n < n_cols
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, n_cols, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, n_cols)
        ranges_k_curr = k + ranges_k
        mask_k = ranges_k_curr < n_cols
        
        # Load weight matrix block
        weight = tl.load(
            weight_ptr + ranges_m * n_cols + ranges_n,
            mask=mask_m & mask_n,
            other=0.0
        )
        
        # Load input matrix block
        input_block = tl.load(
            input_ptr + ranges_k_curr[:, None, None] * (n_batch * n_seq * n_cols) + 
            pid_batch * (n_seq * n_cols) + 
            pid_seq * n_cols + 
            ranges_m[None, None, :] * n_cols + 
            ranges_n[None, :, None],
            mask=mask_k & mask_m[:, None, :] & mask_n[None, :, :],
            other=0.0
        )
        
        # Matrix multiplication
        acc += weight.to(tl.float32) * input_block.to(tl.float32)
    
    # Store result
    tl.store(
        output_ptr + pid_batch * (n_seq * n_rows * n_cols) + 
        pid_seq * (n_rows * n_cols) + 
        ranges_m * n_cols + ranges_n,
        acc,
        mask=mask_m & mask_n
    )

@torch.fx.wrap
def triton_linear(weight, input):
    input_shape = input.shape
    weight_shape = weight.shape
    
    # Calculate output dimensions
    out_features, in_features = weight_shape
    batch, seq_len, _ = input_shape
    
    # Calculate output shape
    out_shape = (batch, seq_len, out_features)
    output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    
    # Block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid calculation
    n_rows = out_features
    n_cols = in_features
    n_batch = batch
    n_seq = seq_len
    
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n, batch, seq_len)
    
    linear_kernel[grid](
        weight,
        input,
        output,
        n_rows,
        n_cols,
        n_batch,
        n_seq,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

@triton.jit
def elementwise_mul_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    batch_size,
    seq_len,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements
    total_elements = batch_size * seq_len * feature_dim
    mask = offsets < total_elements
    
    # Calculate indices
    sample_idx = offsets // (seq_len * feature_dim)
    local_idx = offsets % (seq_len * feature_dim)
    seq_idx = local_idx // feature_dim
    feature_idx = local_idx % feature_dim
    
    # Load data
    a = tl.load(
        a_ptr + sample_idx * seq_len * feature_dim + seq_idx * feature_dim + feature_idx,
        mask=mask,
        other=0.0
    )
    b = tl.load(
        b_ptr + sample_idx * seq_len * feature_dim + seq_idx * feature_dim + feature_idx,
        mask=mask,
        other=0.0
    )
    
    # Element-wise multiplication
    out = a * b
    
    # Store result
    tl.store(
        output_ptr + offsets,
        out,
        mask=mask
    )

@torch.fx.wrap
def triton_elementwise_mul(a, b):
    batch_size, seq_len, feature_dim = a.shape
    output = torch.empty_like(a)
    
    # Calculate total elements
    total_elements = batch_size * seq_len * feature_dim
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    elementwise_mul_kernel[(num_programs,)](
        a,
        b,
        output,
        batch_size,
        seq_len,
        feature_dim,
        BLOCK_SIZE,
    )
    
    return output

@triton.jit
def fused_kernel(
    weight_ptr,
    input_ptr,
    gate_ptr,
    output_ptr,
    n_rows,
    n_cols,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_GATE: tl.constexpr,
):
    # Linear computation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    pid_seq = tl.program_id(3)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    ranges_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    ranges_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    ranges_k = tl.arange(0, BLOCK_SIZE_K)
    
    ranges_m = ranges_m[:, None]
    ranges_n = ranges_n[None, :]
    ranges_k = ranges_k[None, None, :]
    
    mask_m = ranges_m < n_rows
    mask_n = ranges_n < n_cols
    
    linear_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, n_cols, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, n_cols)
        ranges_k_curr = k + ranges_k
        mask_k = ranges_k_curr < n_cols
        
        weight = tl.load(
            weight_ptr + ranges_m * n_cols + ranges_n,
            mask=mask_m & mask_n,
            other=0.0
        )
        
        input_block = tl.load(
            input_ptr + ranges_k_curr[:, None, None] * (batch_size * seq_len * n_cols) +
            pid_batch * (seq_len * n_cols) +
            pid_seq * n_cols +
            ranges_m[None, None, :] * n_cols +
            ranges_n[None, None, :],
            mask=mask_k & mask_m[:, None, :] & mask_n[None, None, :],
            other=0.0
        )
        
        linear_acc += weight.to(tl.float32) * input_block.to(tl.float32)
    
    # Store intermediate result for element-wise multiplication
    tl.store(
        output_ptr + pid_batch * (seq_len * n_rows * n_cols) +
        pid_seq * (n_rows * n_cols) +
        ranges_m * n_cols + ranges_n,
        linear_acc,
        mask=mask_m & mask_n
    )
    
    # Element-wise multiplication computation
    pid_gate = tl.program_id(4)
    gate_start = pid_gate * BLOCK_SIZE_GATE
    gate_offsets = gate_start + tl.arange(0, BLOCK_SIZE_GATE)
    
    total_gate_elements = batch_size * seq_len * n_rows
    mask_gate = gate_offsets < total_gate_elements
    
    # Load linear output for element-wise
    gate_sample_idx = gate_offsets // (seq_len * n_rows)
    gate_local_idx = gate_offsets % (seq_len * n_rows)
    gate_seq_idx = gate_local_idx // n_rows
    gate_feature_idx = gate_local_idx % n_rows
    
    linear_result = tl.load(
        output_ptr + gate_sample_idx * (seq_len * n_rows * n_cols) +
        gate_seq_idx * (n_rows * n_cols) +
        gate_feature_idx * n_cols,
        mask=mask_gate,
        other=0.0
    )
    
    gate_result = tl.load(
        gate_ptr + gate_sample_idx * seq_len * n_rows + gate_seq_idx * n_rows + gate_feature_idx,
        mask=mask_gate & (gate_sample_idx < batch_size) & (gate_seq_idx < seq_len) & (gate_feature_idx < n_rows),
        other=0.0
    )
    
    # Element-wise multiplication
    mul_result = linear_result * gate_result
    
    # Store final result to separate location
    tl.store(
        output_ptr + total_gate_elements + gate_offsets,
        mul_result,
        mask=mask_gate
    )

@torch.fx.wrap
def fused_linear_elementwise(weight, input, gate):
    input_shape = input.shape
    weight_shape = weight.shape
    
    out_features, in_features = weight_shape
    batch, seq_len, _ = input_shape
    
    output_linear = torch.empty((batch, seq_len, out_features), dtype=input.dtype, device=input.device)
    output_mul = torch.empty((batch, seq_len, out_features), dtype=input.dtype, device=input.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_GATE = 1024
    
    grid_m = (out_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (in_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_gate = (batch * seq_len * out_features + BLOCK_SIZE_GATE - 1) // BLOCK_SIZE_GATE
    
    grid = (grid_m, grid_n, batch, seq_len, grid_gate)
    
    # Allocate temporary storage
    temp_storage = torch.empty((batch * seq_len * out_features * 2,), dtype=input.dtype, device=input.device)
    temp_linear = temp_storage[:batch * seq_len * out_features]
    temp_mul = temp_storage[batch * seq_len * out_features:]
    
    fused_kernel[grid](
        weight,
        input,
        gate,
        temp_storage,
        out_features,
        in_features,
        batch,
        seq_len,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_GATE,
    )
    
    # Copy results to proper outputs
    output_linear[:] = temp_storage[:batch * seq_len * out_features].view(batch, seq_len, out_features)
    output_mul[:] = temp_storage[batch * seq_len * out_features:].view(batch, seq_len, out_features)
    
    return (output_mul,)

def replacement_func():
    return fused_linear_elementwise