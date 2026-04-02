import torch
import triton
import triton.language as tl

def pattern(in_5, tmp_1, tmp_0):
    """
    Match the first linear operation with slice + view sequences
    tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_5 = None  # exclude cleanup
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    tmp_4 = None  # exclude cleanup
    """
    tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    return tmp_6, tmp_8

def replacement_args(in_5, tmp_1, tmp_0):
    return (in_5, tmp_1, tmp_0)

@triton.jit
def fused_linear_slice_kernel(
    in_5_ptr, tmp_1_ptr, tmp_0_ptr,
    out_1_ptr, out_2_ptr,
    n_rows, dim_256,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program ID for matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of rows this program should process
    row_start = pid_m * BLOCK_SIZE_M
    row_end = min(row_start + BLOCK_SIZE_M, n_rows)
    rows = row_end - row_start
    
    # Create offsets for matrix A and B
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Mask for row bounds
    mask = row_offsets < n_rows
    
    # Load biases (only first 256 for both outputs)
    bias_256 = tl.load(tmp_0_ptr + tl.arange(0, dim_256), mask=tl.arange(0, dim_256) < dim_256)
    
    # Load the first 256 columns of weights and compute first output
    weights_256 = tl.load(tmp_1_ptr + col_offsets[:, None] + tl.arange(0, dim_256)[None, :],
                        mask=((col_offsets[:, None] + tl.arange(0, dim_256)[None, :]) < (dim_256 * 2)) & 
                              (col_offsets[:, None] < dim_256),
                        other=0.0)
    
    # Load input rows
    input_rows = tl.load(in_5_ptr + row_offsets[:, None] + tl.arange(0, dim_256)[None, :],
                        mask=(row_offsets[:, None] < n_rows[:, None]) & 
                             (tl.arange(0, dim_256)[None, :] < dim_256),
                        other=0.0)
    
    # Compute linear transformation for first 256 columns
    acc = tl.zeros((BLOCK_SIZE_M, dim_256), dtype=tl.float32)
    for k in range(0, dim_256, BLOCK_SIZE_K):
        # Load input chunk
        input_chunk = tl.load(in_5_ptr + row_offsets[:, None] * dim_256 + k + tl.arange(0, BLOCK_SIZE_K)[None, :],
                            mask=(row_offsets[:, None] < n_rows[:, None]) & 
                                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < dim_256),
                            other=0.0)
        
        # Load weights chunk (first 256 columns)  
        weights_chunk = tl.load(tmp_1_ptr + k * 512 + tl.arange(0, BLOCK_SIZE_K)[None, :] + tl.arange(0, dim_256)[:, None],
                              mask=((k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < dim_256) & 
                                   (tl.arange(0, dim_256)[:, None] < dim_256),
                              other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(input_chunk, weights_chunk, out_dtype=tl.float32)
    
    # Add bias
    acc += bias_256
    
    # Store first-half result (first 256 columns)
    out_offsets_1 = row_offsets[:, None] * dim_256 + tl.arange(0, dim_256)[None, :]
    tl.store(out_1_ptr + out_offsets_1, acc, mask=(row_offsets[:, None] < n_rows[:, None]) & 
                                                    (tl.arange(0, dim_256)[None, :] < dim_256))
    
    # Compute and store second-half result (last 256 columns) using the same input
    # Load weights for last 256 columns
    weights_256_last = tl.load(tmp_1_ptr + col_offsets[:, None] + 256 * 512 + tl.arange(0, dim_256)[None, :],
                              mask=((col_offsets[:, None] + 256 * 512 + tl.arange(0, dim_256)[None, :]) < (512 * 512)) & 
                                    (col_offsets + 256 < 512),
                              other=0.0)
    
    # Reuse the same input computation but with different weights
    acc_last = tl.zeros((BLOCK_SIZE_M, dim_256), dtype=tl.float32)
    for k in range(0, dim_256, BLOCK_SIZE_K):
        input_chunk = tl.load(in_5_ptr + row_offsets[:, None] * dim_256 + k + tl.arange(0, BLOCK_SIZE_K)[None, :],
                            mask=(row_offsets[:, None] < n_rows[:, None]) & 
                                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < dim_256),
                            other=0.0)
        
        weights_chunk = tl.load(tmp_1_ptr + k * 512 + 256 * 512 + tl.arange(0, BLOCK_SIZE_K)[None, :] + tl.arange(0, dim_256)[:, None],
                              mask=((k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < dim_256) & 
                                   (tl.arange(0, dim_256)[:, None] + 256 < 512),
                              other=0.0)
        
        acc_last += tl.dot(input_chunk, weights_chunk, out_dtype=tl.float32)
    
    acc_last += bias_256
    
    # Store second-half result (last 256 columns)
    out_offsets_2 = row_offsets[:, None] * dim_256 + tl.arange(0, dim_256)[None, :]
    tl.store(out_2_ptr + out_offsets_2, acc_last, mask=(row_offsets[:, None] < n_rows[:, None]) & 
                                                       (tl.arange(0, dim_256)[None, :] < dim_256))

@torch.fx.wrap
def fused_linear_slice_v1(in_5, tmp_1, tmp_0):
    # Input shapes: in_5 [300, 256], tmp_1 [512, 256], tmp_0 [512]
    n_rows = in_5.shape[0]
    dim_256 = 256
    
    # Output shapes (note: view operations preserve size but reshape internally)
    out_1 = torch.empty((n_rows, dim_256), dtype=in_5.dtype, device=in_5.device)
    out_2 = torch.empty((n_rows, dim_256), dtype=in_5.dtype, device=in_5.device)
    
    # Triton kernel configuration
    BLOCK_SIZE_M = 64  # Process 64 rows at a time
    BLOCK_SIZE_K = 32  # Block size for inner dimension
    
    num_blocks_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_k = 2  # Two separate outputs (first 256 and last 256 columns)
    
    # Launch kernel
    fused_linear_slice_kernel[(num_blocks_m, num_blocks_k)](
        in_5, tmp_1, tmp_0,
        out_1, out_2,
        n_rows, dim_256,
        BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    
    return out_1, out_2

def replacement_func():
    return fused_linear_slice_v1