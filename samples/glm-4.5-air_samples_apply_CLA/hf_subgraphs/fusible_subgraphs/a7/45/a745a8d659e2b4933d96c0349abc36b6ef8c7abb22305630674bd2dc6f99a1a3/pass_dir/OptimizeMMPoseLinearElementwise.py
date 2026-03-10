import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Basic pattern - just linear + multiplication 
    tmp_2 = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return (tmp_3, tmp_2)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

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
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements per sample
    elements_per_sample = seq_len * hidden_size
    
    # Calculate batch and local offsets
    sample_idx = offsets // elements_per_sample
    local_idx = offsets % elements_per_sample
    seq_idx = local_idx // hidden_size
    hidden_idx = local_idx % hidden_size
    
    mask = (sample_idx < batch_size) & (local_idx < elements_per_sample)
    
    # Load data
    a = tl.load(
        a_ptr + sample_idx * elements_per_sample + seq_idx * hidden_size + hidden_idx,
        mask=mask,
        other=0.0
    )
    b = tl.load(
        b_ptr + sample_idx * elements_per_sample + seq_idx * hidden_size + hidden_idx,
        mask=mask,
        other=0.0
    )
    
    # Element-wise multiplication
    out = a * b
    
    # Store result
    tl.store(
        output_ptr + sample_idx * elements_per_sample + seq_idx * hidden_size + hidden_idx,
        out,
        mask=mask
    )

@torch.fx.wrap
def triton_elementwise_mul(a, b):
    # Determine the smaller shape for broadcasting
    if a.shape[0] == 1:
        # Expand to match batch size
        batch_size = b.shape[0]
        seq_len = b.shape[1]
        hidden_size = b.shape[2]
        a_expanded = a.expand(batch_size, seq_len, -1)
    else:
        a_expanded = a
        batch_size = a.shape[0]
        seq_len = a.shape[1]
        hidden_size = a.shape[2]
    
    b_reshaped = b.reshape(batch_size, -1)
    a_reshaped = a_expanded.reshape(batch_size, -1)
    
    output = torch.empty_like(b)
    
    # Calculate total elements
    total_elements = batch_size * seq_len * hidden_size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    elementwise_mul_kernel[(num_programs,)](
        a_reshaped,
        b_reshaped,
        output.reshape(-1),
        batch_size,
        seq_len,
        hidden_size,
        BLOCK_SIZE,
    )
    
    return output

@triton.jit
def fused_kernel(
    weight_ptr,
    input_ptr,
    scale_ptr,
    output_linear_ptr,
    output_mul_ptr,
    n_rows,
    n_cols,
    batch_size,
    seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_ELEM: tl.constexpr,
):
    # Linear computation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    pid_seq = tl.program_id(3)
    
    linear_grid = (pid_m, pid_n, pid_batch, pid_seq)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    ranges_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    ranges_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = ranges_m < n_rows
    mask_n = ranges_n < n_cols
    
    linear_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, n_cols, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, n_cols)
        ranges_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = ranges_k < n_cols
        
        weight = tl.load(
            weight_ptr + ranges_m[:, None] * n_cols + ranges_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0
        )
        
        input_block = tl.load(
            input_ptr + ranges_k[None, None, :] * (batch_size * seq_len * n_cols) +
            pid_batch * (seq_len * n_cols) +
            pid_seq * n_cols +
            ranges_m[None, :, None] * n_cols +
            ranges_n[None, None, :],
            mask=mask_k[None, None, :] & mask_m[None, :, None] & mask_n[None, None, :],
            other=0.0
        )
        
        linear_acc += weight.to(tl.float32) * input_block.to(tl.float32)
    
    # Store linear result
    tl.store(
        output_linear_ptr + pid_batch * (seq_len * n_rows * n_cols) +
        pid_seq * (n_rows * n_cols) +
        ranges_m[:, None] * n_cols + ranges_n[None, :],
        linear_acc,
        mask=mask_m[:, None] & mask_n[None, :]
    )
    
    # Element-wise multiplication computation
    pid_elem = tl.program_id(4)
    elem_start = pid_elem * BLOCK_SIZE_ELEM
    elem_offsets = elem_start + tl.arange(0, BLOCK_SIZE_ELEM)
    
    total_elements = batch_size * seq_len * n_rows
    mask_elem = elem_offsets < total_elements
    
    # Load linear output for element-wise
    sample_idx = elem_offsets // (seq_len * n_rows)
    local_idx = elem_offsets % (seq_len * n_rows)
    seq_idx = local_idx // n_rows
    hidden_idx = local_idx % n_rows
    
    linear_result = tl.load(
        output_linear_ptr + sample_idx * (seq_len * n_rows * n_cols) +
        seq_idx * (n_rows * n_cols) +
        hidden_idx * n_cols,
        mask=mask_elem,
        other=0.0
    )
    
    # Load scale for element-wise
    scale_result = tl.load(
        scale_ptr + sample_idx * seq_len * n_rows + seq_idx * n_rows + hidden_idx,
        mask=mask_elem & (sample_idx < batch_size) & (seq_idx < seq_len) & (hidden_idx < n_rows),
        other=0.0
    )
    
    # Element-wise multiplication
    mul_result = linear_result * scale_result
    
    # Store multiplication result
    tl.store(
        output_mul_ptr + elem_offsets,
        mul_result,
        mask=mask_elem
    )

@torch.fx.wrap
def fused_linear_elementwise(weight, input, scale):
    input_shape = input.shape
    weight_shape = weight.shape
    
    out_features, in_features = weight_shape
    batch, seq_len, _ = input_shape
    
    output_linear = torch.empty((batch, seq_len, out_features), dtype=input.dtype, device=input.device)
    output_mul = torch.empty((batch, seq_len, out_features), dtype=input.dtype, device=input.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_ELEM = 1024
    
    grid_m = (out_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (in_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_elem = (batch * seq_len * out_features + BLOCK_SIZE_ELEM - 1) // BLOCK_SIZE_ELEM
    
    grid = (grid_m, grid_n, batch, seq_len, grid_elem)
    
    fused_kernel[grid](
        weight,
        input,
        scale,
        output_linear,
        output_mul,
        out_features,
        in_features,
        batch,
        seq_len,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_ELEM,
    )
    
    return output_mul, output_linear

def replacement_func():
    return fused_linear_elementwise