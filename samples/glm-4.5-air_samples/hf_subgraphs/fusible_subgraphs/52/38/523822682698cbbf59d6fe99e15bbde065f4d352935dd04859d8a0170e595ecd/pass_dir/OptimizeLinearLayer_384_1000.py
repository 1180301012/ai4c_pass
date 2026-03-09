import torch
import triton
import triton.language as tl

def pattern(in_6, tmp_5, tmp_4):
    tmp_6 = torch.nn.functional.linear(in_6, tmp_5, tmp_4)
    return tmp_6

def replacement_args(in_6, tmp_5, tmp_4):
    return (in_6, tmp_5, tmp_4)

@triton.jit
def linear_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output row (all output features for one sample)
    row = tl.program_id(0)
    M = batch_size
    K = in_features
    N = out_features
    
    # Check if row is valid
    if row >= M:
        return
    
    # Initialize accumulator using scalar approach (avoiding runtime shape issue)
    acc = 0.0
    
    # Compute starting offset for this row
    row_offset = row * K
    
    # Scalar computation for one output element at a time
    for col in range(N):
        # Compute output element offset
        out_col_offset = row * N + col
        
        if out_col_offset < M * N:
            # Initialize scalar accumulator
            acc = 0.0
            
            # Vectorized reduction over input features
            for k in range(0, K, BLOCK_SIZE):
                # Load input chunk
                x_offsets = row_offset + (k + tl.arange(0, BLOCK_SIZE))
                x_mask = (k + tl.arange(0, BLOCK_SIZE)) < K
                x_chunk = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
                
                # Load weight chunk for this column
                w_offsets = k * N + col + tl.arange(0, BLOCK_SIZE)
                w_mask = x_mask  # Reuse input mask for validity
                w_chunk = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0)
                
                # Accumulate dot product
                acc += tl.sum(x_chunk * w_chunk)
            
            # Load bias and add to accumulator
            bias_val = tl.load(bias_ptr + col)
            result = acc + bias_val
            
            # Store result
            tl.store(out_ptr + out_col_offset, result)

@torch.fx.wrap
def triton_linear(x, weight, bias):
    batch_size, in_features = x.shape
    out_features = bias.shape[0]
    
    # Optimal block size for vectorized memory access
    BLOCK_SIZE = 128
    
    # Calculate grid size (one program per row/sample)
    num_programs = batch_size
    
    # Create output tensor
    out = torch.zeros((batch_size, out_features), dtype=x.dtype, device=x.device)
    
    # Launch kernel with optimized block size
    linear_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_linear