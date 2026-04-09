import torch
import triton
import triton.language as tl

# Pattern matching function - matches LayerNorm operation
def pattern(input_tensor, weight, bias):
    """
    Matches the LayerNorm operation exactly as it appears in the computation
    """
    # LayerNorm with parameters
    tmp_2 = torch.nn.functional.layer_norm(input_tensor, (512,), weight, bias, 1e-05)
    return tmp_2

# Arguments extraction function
def replacement_args(input_tensor, weight, bias):
    """
    Extract arguments needed for the optimized kernel
    """
    return (input_tensor, weight, bias)

# Optimized LayerNorm kernel implementation - SIMPLE VERSION
@triton.jit
def optimized_layernorm_kernel(
    # Input tensors
    input_ptr,   # input tensor [1, 3999, 512]
    weight_ptr,  # weight [512]
    bias_ptr,    # bias [512]
    output_ptr,  # output [1, 3999, 512]
    # Tensor dimensions  
    n_batches: tl.constexpr,  # 1
    n_sequence: tl.constexpr,  # 3999
    n_features: tl.constexpr,  # 512
    eps: tl.constexpr,  # 1e-05
    BLOCK_SIZE: tl.constexpr,  # Block size for sequence dimension
):
    """
    Simplified LayerNorm kernel that processes entire feature dimension at once
    for each sequence element in the block
    """
    # Program identifier within launch grid
    seq_idx = tl.program_id(0)  # sequence dimension index
    batch_idx = tl.program_id(1)  # batch dimension index (only 1 batch)
    
    # Calculate sequence offset
    seq_offset = seq_idx * BLOCK_SIZE
    
    # Create bounds checking mask for sequence
    seq_mask = seq_offset + tl.arange(0, BLOCK_SIZE) < n_sequence
    
    # Calculate base address for input and output
    input_base = batch_idx * n_sequence * n_features
    output_base = batch_idx * n_sequence * n_features
    
    # Load weight and bias (entire 512 features at once)
    weight = tl.load(weight_ptr + tl.arange(0, n_features))
    bias = tl.load(bias_ptr + tl.arange(0, n_features))
    
    # Initialize sum and sum of squares for all sequence elements in the block
    local_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    local_sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load and process data for each sequence element in the block
    for seq_pos in range(BLOCK_SIZE):
        if seq_offset + seq_pos < n_sequence:
            # Calculate address for this sequence element
            seq_address = input_base + (seq_offset + seq_pos) * n_features
            
            # Load entire feature vector for this sequence element
            input_features = tl.load(input_ptr + seq_address + tl.arange(0, n_features))
            input_features_fp32 = input_features.to(tl.float32)
            
            # Compute sum and sum of squares for this sequence element
            elem_sum = tl.sum(input_features_fp32)
            elem_sum_sq = tl.sum(input_features_fp32 * input_features_fp32)
            
            # Store in local arrays
            local_sum[seq_pos] = elem_sum
            local_sum_sq[seq_pos] = elem_sum_sq
    
    # Compute mean and variance for all sequence elements in the block
    mean = local_sum * (1.0 / n_features)
    mean_sq = local_sum_sq * (1.0 / n_features)
    variance = mean_sq - mean * mean
    
    # Add epsilon for numerical stability
    std = tl.sqrt(variance + eps)
    
    # Apply LayerNorm and store results
    for seq_pos in range(BLOCK_SIZE):
        if seq_offset + seq_pos < n_sequence:
            # Calculate addresses for input and output
            seq_address = input_base + (seq_offset + seq_pos) * n_features
            output_address = output_base + (seq_offset + seq_pos) * n_features
            
            # Load input features
            input_features = tl.load(input_ptr + seq_address + tl.arange(0, n_features))
            input_features_fp32 = input_features.to(tl.float32)
            
            # LayerNorm: normalize and apply weight/bias
            normalized = ((input_features_fp32 - mean[seq_pos]) / std[seq_pos]) * weight + bias
            
            # Store normalized result
            tl.store(output_ptr + output_address, normalized.to(tl.float16))

# Kernel wrapper - must be decorated with @torch.fx.wrap
@torch.fx.wrap
def optimized_layernorm(input_tensor, weight, bias):
    """
    Wrapper function that sets up and launches the optimized LayerNorm kernel
    """
    # Get tensor shapes
    n_batches = input_tensor.shape[0]  # 1
    n_sequence = input_tensor.shape[1]  # 3999
    n_features = input_tensor.shape[2]  # 512
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Configure block size - optimized for good GPU occupancy
    # Process multiple sequence elements per thread for better utilization
    BLOCK_SIZE = 64  # Number of sequence elements per block
    
    # Calculate grid dimensions (2D grid: sequence and batch)
    grid_seq = (n_sequence + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_batches = n_batches  # Only 1 batch
    
    # Launch the kernel with simplified 2D grid
    optimized_layernorm_kernel[(grid_seq, grid_batches)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_batches=n_batches,
        n_sequence=n_sequence,
        n_features=n_features,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function - returns the optimized kernel function
def replacement_func():
    return optimized_layernorm