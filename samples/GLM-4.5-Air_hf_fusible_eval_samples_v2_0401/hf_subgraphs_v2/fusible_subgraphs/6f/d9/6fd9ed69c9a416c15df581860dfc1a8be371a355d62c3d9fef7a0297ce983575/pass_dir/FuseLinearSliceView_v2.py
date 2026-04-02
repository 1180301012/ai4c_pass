import torch
import triton
import triton.language as tl

def pattern(in_4, tmp_3, tmp_2):
    """
    Match the second linear operation with slice sequences
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
    tmp_9 = tmp_3 = tmp_2 = None  # exclude cleanup
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    tmp_10 = None  # exclude cleanup
    """
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, tmp_3, tmp_2)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    return tmp_11, tmp_12

def replacement_args(in_4, tmp_3, tmp_2):
    return (in_4, tmp_3, tmp_2)

@triton.jit
def fused_linear_slice_v2_kernel(
    in_4_ptr, tmp_3_ptr, tmp_2_ptr,
    out_1_ptr, out_2_ptr,
    n_b, n_s, dim_in, dim_out_full,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program ID for batch processing
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Range of batch elements this program should process
    batch_start = pid_b * BLOCK_SIZE_B
    batch_end = min(batch_start + BLOCK_SIZE_B, n_b)
    batches = batch_end - batch_start
    
    # Create offsets for batch
    batch_offsets = batch_start + tl.arange(0, BLOCK_SIZE_B)
    
    # Mask for batch bounds
    batch_mask = batch_offsets < n_b
    
    # Load biases (only first 256 for both outputs)
    bias_256 = tl.load(tmp_2_ptr + tl.arange(0, 256), mask=tl.arange(0, 256) < 256)
    
    # Load input data for current batch elements
    # Input is [n_b, n_s, dim_in] flattened in memory as [n_b * n_s * dim_in]
    input_data = tl.empty((BLOCK_SIZE_B, n_s, dim_in), dtype=tl.float32)
    for s in range(n_s):
        input_offsets = batch_offsets[:, None] * (n_s * dim_in) + s * dim_in + tl.arange(0, dim_in)[None, :]
        input_slice = tl.load(in_4_ptr + input_offsets,
                            mask=(batch_offsets[:, None] < n_b[:, None]) & 
                                 (tl.arange(0, dim_in)[None, :] < dim_in),
                            other=0.0)
        input_data[:, s, :] = input_slice
    
    # Extract the actual feature data (only the last dimension matters since other dims are size 1)
    # Since this is [300, 1, 256], we just use the last dimension
    features = input_data[:, 0, :]  # [BLOCK_SIZE_B, dim_in]
    
    # Process first 256 columns of output
    if pid_n == 0:  # First 256 columns
        acc_first = tl.zeros((BLOCK_SIZE_B, 256), dtype=tl.float32)
        
        for k in range(0, dim_in, BLOCK_SIZE_K):
            # Load weights chunk (first 256 columns)
            weights_chunk = tl.load(tmp_3_ptr + k * dim_out_full + tl.arange(0, BLOCK_SIZE_K)[None, :] + tl.arange(0, 256)[:, None],
                                  mask=((k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < dim_in) & 
                                       (tl.arange(0, 256)[:, None] < 256),
                                  other=0.0)
            
            # Load input chunk
            input_chunk = features[:, k:k + BLOCK_SIZE_K]
            input_chunk_padded = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_K), dtype=tl.float32)
            input_chunk_padded[:, :min(BLOCK_SIZE_K, dim_in - k)] = input_chunk[:, :min(BLOCK_SIZE_K, dim_in - k)]
            
            # Matrix multiplication
            acc_first += tl.dot(input_chunk_padded, weights_chunk, out_dtype=tl.float32)
        
        # Add bias
        acc_first += bias_256
        
        # Store first-half result
        out_offsets = batch_offsets[:, None] * (n_s * 256) + tl.arange(0, 256)[None, :]
        tl.store(out_1_ptr + out_offsets, acc_first, mask=(batch_offsets[:, None] < n_b[:, None]) & 
                                                        (tl.arange(0, 256)[None, :] < 256))
    
    elif pid_n == 1:  # Last 256 columns
        acc_last = tl.zeros((BLOCK_SIZE_B, 256), dtype=tl.float32)
        
        for k in range(0, dim_in, BLOCK_SIZE_K):
            # Load weights chunk (last 256 columns)
            weights_chunk = tl.load(tmp_3_ptr + k * dim_out_full + 256 * 512 + tl.arange(0, BLOCK_SIZE_K)[None, :] + tl.arange(0, 256)[:, None],
                                  mask=((k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < dim_in) & 
                                       (tl.arange(0, 256)[:, None] + 256 < 512),
                                  other=0.0)
            
            # Load input chunk
            input_chunk = features[:, k:k + BLOCK_SIZE_K]
            input_chunk_padded = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_K), dtype=tl.float32)
            input_chunk_padded[:, :min(BLOCK_SIZE_K, dim_in - k)] = input_chunk[:, :min(BLOCK_SIZE_K, dim_in - k)]
            
            # Matrix multiplication
            acc_last += tl.dot(input_chunk_padded, weights_chunk, out_dtype=tl.float32)
        
        # Add bias (same bias for both halves)
        acc_last += bias_256
        
        # Store second-half result
        out_offsets = batch_offsets[:, None] * (n_s * 256) + tl.arange(0, 256)[None, :]
        tl.store(out_2_ptr + out_offsets, acc_last, mask=(batch_offsets[:, None] < n_b[:, None]) & 
                                                         (tl.arange(0, 256)[None, :] < 256))

@torch.fx.wrap
def fused_linear_slice_v2(in_4, tmp_3, tmp_2):
    # Input shapes: in_4 [1, 150, 1, 512] -> [300, 1, 256] after reshape
    # tmp_3 [512, 256], tmp_2 [512]
    
    # Reshape input to [300, 1, 256] 
    tmp_9 = in_4.reshape(300, -1, 256)
    n_b = tmp_9.shape[0]  # 300
    n_s = tmp_9.shape[1]  # 1  
    dim_in = tmp_9.shape[2]  # 256
    dim_out_full = 512
    
    # Output shapes after slicing
    out_1 = torch.empty((n_b, n_s, 256), dtype=in_4.dtype, device=in_4.device)
    out_2 = torch.empty((n_b, n_s, 256), dtype=in_4.dtype, device=in_4.device)
    
    # Triton kernel configuration
    BLOCK_SIZE_B = 128  # Process 128 batch elements at a time
    BLOCK_SIZE_K = 32   # Block size for inner dimension
    
    num_blocks_b = (n_b + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    num_blocks_n = 2  # Two separate outputs (first 256 and last 256 columns)
    
    # Launch kernel
    fused_linear_slice_v2_kernel[(num_blocks_b, num_blocks_n)](
        tmp_9, tmp_3, tmp_2,
        out_1, out_2,
        n_b, n_s, dim_in, dim_out_full,
        BLOCK_SIZE_B, BLOCK_SIZE_K
    )
    
    return out_1, out_2

def replacement_func():
    return fused_linear_slice_v2