import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_add_relu_kernel(
    bias_ptr,
    weight_ptr,
    input2_ptr,
    input3_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    n_rows: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program id for row blocks
    pid_m = tl.program_id(0)
    # Program id for column blocks  
    pid_k = tl.program_id(1)
    # Offset for rows in output
    row_start = pid_m * BLOCK_SIZE_M
    # Current row range
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    # Mask for valid rows
    row_mask = row_offsets < n_rows
    
    # Load bias vector (shared by all rows in block)
    bias = tl.load(bias_ptr + pid_k, mask=pid_k < n_cols, other=0.0)
    
    # Compute starting column for this block
    col_block_offset = pid_k * BLOCK_SIZE_K
    col_offsets = col_block_offset + tl.arange(0, BLOCK_SIZE_K)
    
    # Load weight matrix block for this column block
    weight_ptrs = weight_ptr + (col_offsets[:, None] * n_cols + tl.arange(0, n_cols)[None, :])
    weight_block = tl.load(weight_ptrs, mask=(col_offsets[:, None] < n_cols) & (tl.arange(0, n_cols)[None, :] < n_cols), other=0.0)
    
    # Load input3 matrix block for row range and column block
    input3_ptrs = input3_ptr + (row_offsets[:, None] * n_cols + col_offsets[None, :])
    input3_block = tl.load(input3_ptrs, mask=(row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols), other=0.0)
    
    # Compute linear transformation: input3 @ weight + bias
    # Reshape for matrix multiplication
    input3_flat = input3_block.reshape(-1, BLOCK_SIZE_K)
    weight_flat = weight_block.reshape(BLOCK_SIZE_K, -1)
    
    # Compute matrix multiplication
    linear_result = tl.dot(input3_flat, weight_flat, out_rows=input3_flat.shape[0], out_cols=weight_flat.shape[1])
    
    # Add bias
    linear_result = linear_result + bias
    
    # Load input2 for addition
    input2_ptrs = input2_ptr + (row_offsets[:, None] * n_cols + tl.arange(0, n_cols)[None, :])
    input2_block = tl.load(input2_ptrs, mask=row_mask[:, None], other=0.0)
    
    # Add input2
    result = linear_result + input2_block
    
    # Apply ReLU activation
    result = tl.maximum(result, 0.0)
    
    # Store output
    output_ptrs = output_ptr + (row_offsets[:, None] * n_cols + tl.arange(0, n_cols)[None, :])
    tl.store(output_ptrs, result, mask=(row_offsets[:, None] < n_rows) & (tl.arange(0, n_cols)[None, :] < n_cols))



@triton.jit
def fused_linear_add_relu_kernel(
    bias_ptr,
    weight_ptr, 
    input2_ptr,
    input3_ptr,
    output_ptr,
    tensor_dtype: tl.constexpr,
    n_cols: tl.constexpr,
    n_rows: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr
):
    # Program id for row blocks
    pid = tl.program_id(0)
    
    # Number of programs needed
    n_programs = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    if pid >= n_programs:
        return
        
    # Row range for this program  
    row_start = pid * BLOCK_SIZE_M
    row_offset = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offset < n_rows
    
    # Load bias
    bias = tl.load(bias_ptr, mask=True, other=0.0)
    
    # Compute: for each row i and column j: sum_k(input3[i,k] * weight[k,j]) + bias[j]
    # Initialize result matrix with zeros using passed dtype
    result = tl.zeros((BLOCK_SIZE_M, n_cols), dtype=tensor_dtype)
    
    # Compute matrix multiplication: input3 @ weight.T + bias
    for k in range(0, n_cols):
        # Load k-th column of weights (this will be the j-th column in result)
        weight_ptrs = weight_ptr + k * n_cols + tl.arange(0, n_cols)
        weight_col = tl.load(weight_ptrs, mask=tl.arange(0, n_cols) < n_cols, other=0.0)
        
        # Load k-th element from each input3 row (column vector)
        input3_ptrs = input3_ptr + row_offset * n_cols + k
        input3_col = tl.load(input3_ptrs, mask=row_mask, other=0.0)
        
        # Outer product: input3_col[BLOCK_SIZE_M] * weight_col[n_cols] -> [BLOCK_SIZE_M, n_cols]
        contribution = input3_col[:, None] * weight_col[None, :]
        
        # Add to result
        result += contribution
    
    # Add bias broadcast across all rows
    result += bias[None, :]
    
    # Load input2 and add it
    input2_ptrs = input2_ptr + row_offset[:, None] * n_cols + tl.arange(0, n_cols)[None, :]
    input2_block = tl.load(input2_ptrs, mask=(row_mask[:, None]) & (tl.arange(0, n_cols)[None, :] < n_cols), other=0.0)
    result += input2_block
    
    # Apply ReLU activation
    result = tl.maximum(result, 0.0)
    
    # Store output
    output_ptrs = output_ptr + row_offset[:, None] * n_cols +tl.arange(0, n_cols)[None, :]
    tl.store(output_ptrs, result, mask=(row_mask[:, None]) & (tl.arange(0, n_cols)[None, :] < n_cols))

@torch.fx.wrap
def fused_linear_add_relu(in_0, in_1, in_2, in_3):
    input_shape = in_3.shape
    n_rows, n_cols = input_shape
    
    # Map PyTorch dtype to Triton dtype
    if in_3.dtype == torch.float32:
        tensor_dtype = tl.float32
    elif in_3.dtype == torch.float16:
        tensor_dtype = tl.float16
    elif in_3.dtype == torch.bfloat16:
        tensor_dtype = tl.bfloat16
    else:
        tensor_dtype = tl.float16  # fallback
    
    # Use power-of-2 BLOCK_SIZE for compatibility with Triton requirements
    if n_rows > 1024:
        BLOCK_SIZE_M = 256
    elif n_rows > 512:
        BLOCK_SIZE_M = 128  
    elif n_rows > 256:
        BLOCK_SIZE_M = 64
    else:
        BLOCK_SIZE_M = 32
    
    # Create output tensor
    out = torch.empty_like(in_3)
    
    # Grid dimensions (only row blocks needed)
    grid_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    fused_linear_add_relu_kernel[(grid_m,)](
        bias_ptr=in_0,
        weight_ptr=in_1, 
        input2_ptr=in_2,
        input3_ptr=in_3,
        output_ptr=out,
        tensor_dtype=tensor_dtype,
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return out

def replacement_func():
    return fused_linear_add_relu