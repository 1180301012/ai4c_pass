import torch
import triton
import triton.language as tl

# Pattern 1: Linear transformation on in_5 with slicing and reshaping
def pattern(in_5, in_1, in_0):
    tmp_4 = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5 = tmp_4[:, :256]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[:, -256:]
    tmp_8 = tmp_7.view(-1, 256)
    return tmp_6, tmp_8

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def linear_slice_kernel1(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_first_ptr, 
    out_second_ptr,
    n_rows,
    n_cols_output,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles a block of rows
    row_start = tl.program_id(0) * BLOCK_SIZE_M
    row_end = min(row_start + BLOCK_SIZE_M, n_rows)
    
    # Allocate shared memory for accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(n_cols_output, BLOCK_SIZE_K)):
        # Load bias for this block
        bias_idx = row_start + tl.arange(0, row_end - row_start)
        bias = tl.load(bias_ptr + bias_idx, mask=bias_idx < n_cols_output, other=0.0).to(tl.float32)
        
        # Loop over N dimension in blocks
        for n in range(0, tl.cdiv(256, BLOCK_SIZE_N)):
            col_start = n * BLOCK_SIZE_N
            col_end = min(col_start + BLOCK_SIZE_N, 256)
            
            # Load input data
            x_cols = tl.arange(col_start, col_end)
            x_mask = x_cols < 256
            x_data = tl.load(x_ptr + row_start * 256 + x_cols, mask=x_mask, other=0.0).to(tl.float32)
            
            # Load weight data for first half (first 256 columns)
            weight_cols = tl.arange(col_start, col_end)
            weight_mask = weight_cols < 256
            weight_data_first = tl.load(weight_ptr + weight_cols * 512 + 0, mask=weight_mask, other=0.0).to(tl.float32)
            
            # Load weight data for second half (last 256 columns)  
            weight_data_second = tl.load(weight_ptr + weight_cols * 512 + 256, mask=weight_mask, other=0.0).to(tl.float32)
            
            # Matrix multiplication for first half
            for k_inner in range(0, tl.cdiv(256, BLOCK_SIZE_K)):
                k_start = k_inner * BLOCK_SIZE_K
                k_end = min(k_start + BLOCK_SIZE_K, 256)
                k_idx = tl.arange(k_start, k_end)
                k_mask = k_idx < 256
                
                # Load weights from different K blocks
                weight_k_first = tl.load(weight_ptr + weight_data_first + k_idx * 512 + tl.arange(0, 256), 
                                       mask=k_mask.outer(weight_mask), other=0.0)
                weight_k_second = tl.load(weight_ptr + weight_data_second + k_idx * 512 + tl.arange(0, 256), 
                                        mask=k_mask.outer(weight_mask), other=0.0)
                
                # Compute and accumulate
                acc_first = tl.dot(x_data, weight_k_first.to(tl.float32))
                acc_second = tl.dot(x_data, weight_k_second.to(tl.float32))
                
                if n == 0 and k == 0:
                    accumulator = tl.reshape(tl.stack([acc_first, acc_second], dim=1), 
                                           (row_end - row_start, 256))
    
    # Add bias and store results
    output_first = accumulator[:, :256] + bias.unsqueeze(0)
    output_second = accumulator[:, 256:] + bias.unsqueeze(0)
    
    # Store first half
    first_cols = tl.arange(0, 256)
    first_mask = row_start + tl.arange(0, row_end - row_start) < n_rows
    tl.store(out_first_ptr + (row_start * 256 + first_cols).unsqueeze(0), 
             output_first, mask=first_mask.outer(tl.arange(0, 256) < 256))
    
    # Store second half  
    second_cols = tl.arange(0, 256)
    tl.store(out_second_ptr + (row_start * 256 + second_cols).unsqueeze(0), 
             output_second, mask=first_mask.outer(tl.arange(0, 256) < 256))

@torch.fx.wrap
def optimized_linear_slice1(in_5, in_1, in_0):
    n_rows = in_5.shape[0]  # 300
    n_cols_output = in_0.shape[0]  # 512
    
    # Output tensors for both halves
    out_first = torch.empty((n_rows, 256), device=in_5.device, dtype=torch.float32)
    out_second = torch.empty((n_rows, 256), device=in_5.device, dtype=torch.float32)
    
    # Grid configuration
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    num_programs = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    linear_slice_kernel1[(num_programs,)](
        in_5,
        in_1,
        in_0,
        out_first,
        out_second,
        n_rows,
        n_cols_output,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return out_first, out_second

def replacement_func():
    return optimized_linear_slice1