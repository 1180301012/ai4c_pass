import torch
import triton
import triton.language as tl

def pattern(tmp_9, in_3, in_2):
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = linear_1[..., :256]
    tmp_12 = linear_1[..., -256:]
    return tmp_11, tmp_12

def replacement_args(tmp_9, in_3, in_2):
    return (tmp_9, in_3, in_2)

@triton.jit
def linear_split_kernel(
    x_ptr,          # tmp_9: [300, 1, 256]
    w_ptr,          # in_3: [512, 256] (transposed weights)
    b_ptr,          # in_2: [512]
    out_first_ptr,  # tmp_11: [300, 1, 256] (first half of linear output)
    out_second_ptr, # tmp_12: [300, 1, 256] (second half of linear output)
    n_rows,
    n_inner,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program ID for row partitioning
    pid_m = tl.program_id(0)
    
    # Range of rows this program should process
    row_start = pid_m * BLOCK_SIZE_M
    row_end = min((pid_m + 1) * BLOCK_SIZE_M, n_rows)
    rows = row_end - row_start
    
    # Load full bias
    bias = tl.load(b_ptr + tl.arange(0, 512), mask=tl.arange(0, 512) < 512, other=0.0)
    
    # Initialize output accumulators
    first_half = tl.zeros((rows, n_inner, 256), dtype=tl.float32)
    second_half = tl.zeros((rows, n_inner, 256), dtype=tl.float32)
    
    # Process input in chunks to avoid memory issues
    for k in range(0, 256, 32):  # Process in 32-element chunks
        chunk_size = min(32, 256 - k)
        
        # Load input chunk
        x_offset = row_start * n_inner * 256 + k * n_inner
        x_ptrs = x_offset + tl.arange(0, rows * n_inner * chunk_size)
        x_chunk = tl.load(x_ptrs, mask=tl.arange(0, rows * n_inner * chunk_size) < rows * n_inner * 256, other=0.0)
        x_chunk = x_chunk.reshape(rows, n_inner, chunk_size)
        
        # Load corresponding weights for first half
        w_first_offset = k * 256
        w_first_ptrs = w_first_offset + tl.arange(0, chunk_size * 256).reshape(chunk_size, 256)
        w_first = tl.load(w_first_ptrs, mask=(tl.arange(0, chunk_size)[:, None] < chunk_size) & (tl.arange(0, chunk_size)[:, None] < 256), other=0.0)
        
        # Compute first half contribution
        if chunk_size == 256:
            x_flat = x_chunk.reshape(rows * n_inner, 256)
            result_first = tl.dot(x_flat, w_first, out_type=tl.float32)
            result_first = result_first.reshape(rows, n_inner, 256)
            first_half += result_first
            
            # Load corresponding weights for second half  
            w_second_offset = k * 256 + 256 * 256  # Second half of weight matrix
            w_second_ptrs = w_second_offset + tl.arange(0, chunk_size * 256).reshape(chunk_size, 256)
            w_second = tl.load(w_second_ptrs, mask=(tl.arange(0, chunk_size)[:, None] < chunk_size) & (tl.arange(0, chunk_size)[:, None] < 256), other=0.0)
            
            result_second = tl.dot(x_flat, w_second, out_type=tl.float32)
            result_second = result_second.reshape(rows, n_inner, 256)
            second_half += result_second
        else:
            # For smaller chunks, accumulate
            x_flat = x_chunk.reshape(rows * n_inner, chunk_size)
            
            # First half weights
            w_first_block = w_first[:, :256]
            result_first = tl.dot(x_flat, w_first_block, out_type=tl.float32)
            result_first = result_first.reshape(rows, n_inner, 256)
            first_half += result_first
            
            # Second half weights
            w_second_block = w_first[:, 256:]
            result_second = tl.dot(x_flat, w_second_block, out_type=tl.float32)
            result_second = result_second.reshape(rows, n_inner, 256)
            second_half += result_second
    
    # Add bias
    bias_first = bias[:256].reshape(1, 1, 256)
    bias_second = bias[256:].reshape(1, 1, 256)
    
    final_first = first_half + bias_first
    final_second = second_half + bias_second
    
    # Store results
    first_offset = row_start * n_inner * 256
    second_offset = row_start * n_inner * 256
    
    tl.store(out_first_ptr + first_offset, final_first, mask=tl.arange(0, rows * n_inner * 256) < rows * n_inner * 256)
    tl.store(out_second_ptr + second_offset, final_second, mask=tl.arange(0, rows * n_inner * 256) < rows * n_inner * 256)

@torch.fx.wrap  
def fused_linear_slice_split(tmp_9, in_3, in_2):
    n_rows = tmp_9.shape[0]
    n_inner = tmp_9.shape[1]  # This should be 1 based on the reshape
    
    # Output tensors
    out_first = torch.empty((n_rows, n_inner, 256), dtype=tmp_9.dtype, device=tmp_9.device)
    out_second = torch.empty((n_rows, n_inner, 256), dtype=tmp_9.dtype, device=tmp_9.device)
    
    # Launch kernel
    grid = (lambda meta: (
        (n_rows + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
    ))
    
    linear_split_kernel[grid](
        tmp_9,
        in_3,
        in_2,
        out_first,
        out_second,
        n_rows,
        n_inner,
        BLOCK_SIZE_M=32,  # 32 rows per block
    )
    
    return out_first, out_second

def replacement_func():
    return fused_linear_slice_split