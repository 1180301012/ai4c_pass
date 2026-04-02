import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """Match linear + reshape + softmax computation pattern"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Kernel that handles linear transformation + reshape
@triton.jit
def linear_reshape_kernel(
    x_ptr,           # input tensor [1, 19, 128]
    w_ptr,           # weight [18, 128]  
    b_ptr,           # bias [18]
    out_ptr,         # output [38, 9] (flattened from [38, 9, 1])
    n_batch: tl.constexpr,     # batch size = 1
    n_seq: tl.constexpr,       # sequence length = 19  
    n_in: tl.constexpr,        # input dim = 128
    n_out: tl.constexpr,       # output dim = 18
    reshape_n: tl.constexpr,   # reshape middle dim = 9
):
    # Each program handles one output element in final [38, 9] format
    
    # Total elements in original [1, 19, 18] -> 342
    # Total elements in final [38, 9] -> 342
    pid = tl.program_id(0)
    
    # Determine linear indices in original [1, 19, 18] -> [342] format
    # pid here goes from 0 to 341 corresponding to flattened [1, 19, 18]
    if pid >= n_seq * n_out:  # 19 * 18 = 342
        return
    
    # Original indices in [1, 19, 18]
    m = pid // n_out        # 0 to 18 (sequence index, batch is always 0)
    n = pid % n_out         # 0 to 17 (feature index)
    
    # Compute new indices in [38, 9] format
    # Reshape: [1, 19, 18] -> [38, 9, 1]
    # Each original [m, n] where m=0..18, n=0..17 becomes:
    # Since we want [38, 9], we split the 18 features into 2 groups of 9
    # row = m * 2 + (n // 9)   # Each original row becomes 2 rows in output
    # col = n % 9              # First 9 features go to first row, next 9 to second row
    new_row = m * 2 + (n // reshape_n)   # m * 2 + (n // 9)
    new_col = n % reshape_n               # n % 9
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input dimension (K) for matrix multiplication
    for k in range(0, n_in, 32):  # Process input in chunks of 32
        k_end = tl.minimum(k + 32, n_in)
        k_mask = k < k_end
        
        # Load input [m, k] where m=0..18
        x = tl.load(
            x_ptr + m * n_in + k,
            mask=k_mask,
            other=0.0
        ).to(tl.float32)
        
        # Load weight [n, k] where n=0..17
        w = tl.load(
            w_ptr + n * n_in + k,
            mask=k_mask,
            other=0.0
        ).to(tl.float32)
        
        # Matrix multiplication element
        acc += x * w
    
    # Load bias and add
    bias = tl.load(b_ptr + n).to(tl.float32)
    acc += bias
    
    # Convert back to original dtype and store in [38, 9] format
    tl.store(out_ptr + new_row * reshape_n + new_col, acc.to(tl.bfloat16))

# Softmax kernel
@triton.jit
def softmax_kernel(
    in_ptr,          # input [38, 9] (linear results, flattened)
    out_ptr,         # output [38, 9] (softmax results, flattened)
    n_rows: tl.constexpr,      # number of rows = 38
    n_cols: tl.constexpr,      # number of columns = 9
):
    # Each program computes one row of softmax
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Load the entire row using power-of-2 vector size
    row_offset = row_id * n_cols
    row_indices = tl.arange(0, 8)  # Use 8 which is a power of 2
    row_mask = row_indices < n_cols
    
    # Load first 8 elements
    row_data = tl.load(
        in_ptr + row_offset + row_indices,
        mask=row_mask,
        other=0.0
    ).to(tl.float32)
    
    # Compute max for numerical stability
    row_max = tl.max(row_data)
    
    # Compute exp and sum for first 8 elements
    row_exp = tl.exp(row_data - row_max)
    row_sum = tl.sum(row_exp)
    
    # Handle the 9th element if needed
    if n_cols > 8:
        # Load the 9th element separately
        last_element = tl.load(
            in_ptr + row_offset + 8,
            mask=8 < n_cols,
            other=0.0
        ).to(tl.float32)
        
        # Include it in the computation
        last_exp = tl.exp(last_element - row_max)
        row_sum += last_exp
        
        # Store 9th element softmax result
        last_softmax = last_exp / (row_sum + 1e-20)
        tl.store(
            out_ptr + row_offset + 8,
            last_softmax.to(tl.bfloat16),
            mask=8 < n_cols
        )
    
    # Compute softmax and store first 8 elements
    row_softmax = row_exp / (row_sum + 1e-20)  # add epsilon for stability
    tl.store(
        out_ptr + row_offset + row_indices,
        row_softmax.to(tl.bfloat16),
        mask=row_mask
    )

# Kernel wrapper
@torch.fx.wrap
def fused_linear_reshape_softmax(in_0, in_1, in_2):
    # Get tensor properties
    batch_size, seq_len, in_features = in_2.shape  # [1, 19, 128]
    out_features = in_1.shape[0]  # 18
    
    # Create output tensor [38, 9] (flattened from [38, 9, 1])
    reshape_n = 9
    output_rows = (batch_size * seq_len * out_features) // reshape_n  # 38
    linear_reshape_out = torch.empty((output_rows, reshape_n), 
                                   dtype=in_2.dtype, device=in_2.device)
    
    # Launch linear transformation + reshape kernel
    total_elements = seq_len * out_features  # 19 * 18 = 342
    grid_lambda = lambda meta: (total_elements,)
    
    linear_reshape_kernel[grid_lambda](
        in_2, in_1, in_0, linear_reshape_out,
        batch_size, seq_len, in_features, out_features, reshape_n
    )
    
    # Create temporary buffer for softmax input and output
    # Softmax operates on [38, 9] tensor - apply softmax along dimension 1
    softmax_input = torch.empty((output_rows, reshape_n), 
                               dtype=torch.float32, device=in_2.device)
    softmax_output = torch.empty((output_rows, reshape_n), 
                                dtype=in_2.dtype, device=in_2.device)
    
    # Copy linear_reshape output to softmax input, converting to float32
    softmax_input.copy_(linear_reshape_out.to(torch.float32))
    
    # Launch softmax kernel
    softmax_grid = lambda meta: (output_rows,)
    softmax_kernel[softmax_grid](
        softmax_input, softmax_output,
        output_rows, reshape_n
    )
    
    # Reshape from [38, 9] to [38, 9, 1] to match expected output shape
    final_output = softmax_output.unsqueeze(-1)
    return final_output

# Replacement function
def replacement_func():
    return fused_linear_reshape_softmax