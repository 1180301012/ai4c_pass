import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor):
    """Pattern for mean reduction along dimension -2"""
    result = input_tensor.mean(-2)
    return result

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for optimized mean reduction
@triton.jit
def mean_kernel(
    input_ptr, input_strides,
    output_ptr, output_stride,
    n_batch, n_seq, n_features,
    BLOCK_SIZE_N: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Determine which batch and feature this program handles
    batch_id = pid // n_features
    feature_id = pid % n_features
    
    # Compute output position
    output_pos = batch_id * output_stride + feature_id
    
    # Initialize sum accumulator
    sum_val = 0.0
    
    # Loop over sequence dimension (dimension -2)
    for seq_pos in range(n_seq):
        # Compute input position: [batch_id, seq_pos, feature_id]
        input_pos = (batch_id * input_strides[0] + 
                    seq_pos * input_strides[1] + 
                    feature_id)
        
        # Load element
        val = tl.load(input_ptr + input_pos, other=0.0)
        sum_val += val
    
    # Compute mean
    mean_val = sum_val / n_seq
    
    # Store result
    tl.store(output_ptr + output_pos, mean_val)

# Alternative optimized kernel using shared memory for better performance
@triton.jit
def mean_kernel_optimized(
    input_ptr, input_strides,
    output_ptr, output_stride,
    n_batch, n_seq, n_features,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEATURES: tl.constexpr, BLOCK_SIZE_SEQ: tl.constexpr
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    
    # Check bounds
    if pid_batch >= n_batch or pid_feature >= n_features:
        return
    
    # Shared memory for partial sums
    shared_mem = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_FEATURES), dtype=tl.float32)
    
    # Compute block start positions
    batch_start = pid_batch * BLOCK_SIZE_BATCH
    feature_start = pid_feature * BLOCK_SIZE_FEATURES
    
    # Accumulate partial sums using shared memory
    total_sum = 0.0
    element_count = 0
    
    # Loop over sequence dimension with shared memory tiling
    for seq_pos in range(n_seq):
        for block_batch in range(BLOCK_SIZE_BATCH):
            for block_feature in range(BLOCK_SIZE_FEATURES):
                batch_id = batch_start + block_batch
                feature_id = feature_start + block_feature
                
                if batch_id < n_batch and feature_id < n_features:
                    # Compute input position
                    input_pos = (batch_id * input_strides[0] + 
                               seq_pos * input_strides[1] + 
                               feature_id)
                    
                    # Load element
                    val = tl.load(input_ptr + input_pos, other=0.0)
                    total_sum += val
                    element_count += 1
    
    # Compute mean
    if element_count > 0:
        mean_val = total_sum / element_count
    else:
        mean_val = 0.0
    
    # Store result
    output_pos = pid_batch * output_stride + pid_feature
    tl.store(output_ptr + output_pos, mean_val)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_mean_reduction(input_tensor):
    # Get tensor shape
    shape = input_tensor.shape
    n_batch = shape[0]
    n_seq = shape[1]  # This is the dimension being reduced (-2)
    n_features = shape[2]
    
    # Create output tensor
    output_shape = (n_batch, n_features)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Choose kernel strategy based on problem size
    if n_seq >= 32 and n_features >= 32 and n_batch >= 32:
        # Use the optimized kernel for larger problems
        BLOCK_SIZE_BATCH = 8
        BLOCK_SIZE_FEATURES = 8
        BLOCK_SIZE_SEQ = 32
        
        grid_batch = (n_batch + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH
        grid_features = (n_features + BLOCK_SIZE_FEATURES - 1) // BLOCK_SIZE_FEATURES
        
        mean_kernel_optimized[(
            grid_batch * grid_features,
        )](
            input_ptr=input_tensor,
            input_strides=[input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2)],
            output_ptr=output,
            output_stride=output.stride(0),
            n_batch=n_batch,
            n_seq=n_seq,
            n_features=n_features,
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_FEATURES=BLOCK_SIZE_FEATURES,
            BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ
        )
    else:
        # Use simpler kernel for smaller problems
        BLOCK_SIZE_N = 32
        
        grid_size = n_batch * n_features
        mean_kernel[(
            grid_size,
        )](
            input_ptr=input_tensor,
            input_strides=[input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2)],
            output_ptr=output,
            output_stride=output.stride(0),
            n_batch=n_batch,
            n_seq=n_seq,
            n_features=n_features,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_mean_reduction