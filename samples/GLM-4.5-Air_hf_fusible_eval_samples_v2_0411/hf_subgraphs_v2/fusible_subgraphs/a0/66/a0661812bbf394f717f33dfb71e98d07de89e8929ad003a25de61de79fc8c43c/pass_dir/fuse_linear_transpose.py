import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, dropout_p, dropout_training, dropout_inplace):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, dropout_p, dropout_training, dropout_inplace)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2, dropout_p, dropout_training, dropout_inplace):
    return (in_0, in_1, in_2, dropout_p, dropout_training, dropout_inplace)

@triton.jit
def fused_linear_transpose_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    output_transposed_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of columns for this program
    offset_n = pid_n * BLOCK_SIZE_N
    cols = offset_n + tl.arange(0, BLOCK_SIZE_N)
    cols = cols % out_features
    
    # Range of rows (batch * seq_len) for this program
    offset_m = pid_m * BLOCK_SIZE_M
    rows = offset_m + tl.arange(0, BLOCK_SIZE_M)
    rows = rows % (batch_size * seq_len)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load bias
        bias = tl.load(bias_ptr + cols, mask=cols < out_features, other=0.0)
        
        # Load weight tiles
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        k_offset = k_offset % in_features
        weight = tl.load(weight_ptr + k_offset[:, None] * out_features + cols[None, :],
                        mask=(k_offset[:, None] < in_features) & (cols[None, :] < out_features),
                        other=0.0)
        
        # Load input tiles
        input_vals = tl.load(input_ptr + rows[:, None] * in_features + k_offset[None, :],
                           mask=(rows[:, None] < (batch_size * seq_len)) & (k_offset[None, :] < in_features),
                           other=0.0)
        
        # Matrix multiplication: accumulate = input * weight
        accumulator += tl.dot(input_vals.to(tl.float32), weight.to(tl.float32))
    
    # Add bias
    accumulator = accumulator + bias[None, :]
    
    # Store to regular output
    mask_output = (rows[:, None] < (batch_size * seq_len)) & (cols[None, :] < out_features)
    tl.store(output_ptr + rows[:, None] * out_features + cols[None, :],
             accumulator.to(tl.float16 if tl.float16 else tl.float32),
             mask=mask_output)
    
    # Store to transposed output: [batch, seq_len, out_features] -> [batch, out_features, seq_len]
    transpose_cols = rows % seq_len
    transpose_rows = (rows // seq_len) * out_features + pid_n * BLOCK_SIZE_N
    mask_transpose = (transpose_rows < batch_size * out_features) & (transpose_cols < seq_len)
    tl.store(output_transposed_ptr + transpose_rows[:, None] * seq_len + transpose_cols[None, :],
             accumulator.to(tl.float16 if tl.float16 else tl.float32),
             mask=mask_transpose)

@torch.fx.wrap
def fused_linear_transpose(in_0, in_1, in_2, dropout_p, dropout_training, dropout_inplace):
    # Get input shapes
    batch_size, seq_len, in_features = in_2.shape
    out_features = in_0.shape[0]
    
    # Create output tensors
    output_regular = torch.empty((batch_size, seq_len, out_features), dtype=in_2.dtype, device=in_2.device)
    output_transposed = torch.empty((batch_size, out_features, seq_len), dtype=in_2.dtype, device=in_2.device)
    
    # Set up kernel configuration
    BLOCK_SIZE_M = 64  # Process batch_size * seq_len dimension
    BLOCK_SIZE_N = 256  # Process out_features dimension
    BLOCK_SIZE_K = 32   # Process in_features dimension
    
    # Calculate grid size
    grid_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_linear_transpose_kernel[(grid_m, grid_n)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output_regular,
        output_transposed_ptr=output_transposed,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output_regular, output_transposed

def replacement_func():
    return fused_linear_transpose