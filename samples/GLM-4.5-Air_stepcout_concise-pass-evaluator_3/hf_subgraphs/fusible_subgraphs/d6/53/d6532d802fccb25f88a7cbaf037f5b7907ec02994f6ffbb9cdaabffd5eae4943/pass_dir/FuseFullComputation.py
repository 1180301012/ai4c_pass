import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the complete computation pattern from the original model
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return (tmp_6, tmp_17, tmp_7)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_trigonometric_kernel(
    x_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both cos and sin simultaneously
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store both results
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)

@triton.jit
def fused_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    D,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 2D tensor: (N, D)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block bounds
    m_begin = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, N)
    n_begin = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, D)
    
    # Accumulators for mean
    sum_x = 0.0
    
    # Process the block
    for j in range(n_begin, n_end):
        # Get pointer for this column, all rows
        col_ptr = input_ptr + j
        # Load a column and compute sum
        row_start = m_begin
        row_end = m_end
        for i in range(row_start, row_end):
            offset = i * D + j
            val = tl.load(input_ptr + offset, other=0.0)
            sum_x += val * val
    
    # Sum across all programs for this column
    block_sum = tl.sum(sum_x, axis=0)
    
    # Mean and variance
    mean = tl.math.rsqrt(block_sum / N + eps)
    
    # Apply normalization
    for j in range(n_begin, n_end):
        col_ptr = input_ptr + j
        for i in range(m_begin, m_end):
            offset = i * D + j
            x = tl.load(input_ptr + offset, other=0.0)
            normalized_x = x * mean
            tl.store(output_ptr + offset, normalized_x)

@torch.fx.wrap
def fused_trigonometric_ops(in_1):
    # Determine input shape after concatenation
    original_shape = in_1.shape
    concated_shape = (*original_shape[:-1], original_shape[-1] * 2)
    
    # Reshape if needed for contiguous memory access
    if in_1.stride() != tuple(range(len(in_1.shape)-1, -1, -1)):
        in_1 = in_1.contiguous()
    
    # Compute total elements and create output tensors
    n_elements = 1
    for dim in concated_shape:
        n_elements *= dim
    
    cos_output = torch.empty(concated_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_output = torch.empty(concated_shape, dtype=torch.bfloat16, device=in_1.device)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_trigonometric_kernel[(num_programs,)](
        in_1.flatten(),
        cos_output.flatten(),
        sin_output.flatten(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_output, sin_output

@torch.fx.wrap 
def fused_layer_norm_with_mul(in_2, in_0):
    # Convert to float32 for computation
    x_fp32 = in_2.to(torch.float32)
    
    # Get tensor dimensions
    if len(x_fp32.shape) == 3:
        # Shape: [batch, seq_len, hidden_dim]
        N = x_fp32.shape[0] * x_fp32.shape[1]
        D = x_fp32.shape[2]
    else:
        raise ValueError(f"Unexpected input shape: {x_fp32.shape}")
    
    # Ensure contiguous memory
    x_fp32 = x_fp32.contiguous()
    
    # Create output tensor 
    output = torch.empty_like(in_2)  # Will store the final bfloat16 result
    
    # Layer normalization parameters
    eps = 1e-06
    
    # Launch kernels with appropriate tiling for GPU
    BLOCK_SIZE_M = 64   # Process 64 rows at a time
    BLOCK_SIZE_N = 128  # Process 128 columns at a time
    
    # Check if we need to adjust grid size based on actual dimensions
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (D + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Handle small case with simpler kernel
    if N <= 1024 and D <= 1024:
        # Use vectorized processing for small tensors
        BLOCK_SIZE = min(1024, D)
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        @triton.jit
        def simple_layer_norm_kernel(
            x_ptr,
            out_ptr,
            N,
            D,
            eps: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            row_start = pid * BLOCK_SIZE
            row_end = min((pid + 1) * BLOCK_SIZE, N)
            
            for i in range(row_start, row_end):
                # Compute mean for this row
                sum_x = 0.0
                for j in range(D):
                    offset = i * D + j
                    x = tl.load(x_ptr + offset, other=0.0)
                    sum_x += x * x
                
                mean = tl.math.rsqrt(sum_x / D + eps)
                
                # Store normalized values
                for j in range(D):
                    offset = i * D + j
                    x = tl.load(x_ptr + offset, other=0.0)
                    normalized_x = x * mean
                    tl.store(out_ptr + offset, normalized_x)
        
        simple_layer_norm_kernel[(num_programs,)](
            x_fp32.flatten(),
            output.flatten(),
            N, D, eps, BLOCK_SIZE
        )
    else:
        # Use the 2D grid kernel for larger tensors
        fused_layer_norm_kernel[(grid_m, grid_n)](
            x_fp32.flatten(),
            in_0.flatten(),  # weight (broadcasted)
            output.flatten(),
            N, D, eps,
            BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    
    return output

def replacement_func():
    def full_fused_computation(in_0, in_1, in_2):
        # Step 1: Fused trigonometric operations
        cos_output, sin_output = fused_trigonometric_ops(in_1)
        
        # Step 2: Fused layer normalization with final multiplication
        normalized_output = fused_layer_norm_with_mul(in_2, in_0)
        final_output = in_0 * normalized_output
        
        return (cos_output, final_output, sin_output)
    
    return full_fused_computation