import torch
import triton
import triton.language as tl

@triton.jit
def linear_mul_kernel_fused(
    x_ptr,           # Input tensor for linear: [batch, seq_len, in_features]
    weight_ptr,      # Weight matrix: [out_features, in_features]
    scale_ptr,       # Scale vector: [out_features] 
    out_ptr,         # Output tensor: [batch, seq_len, out_features]
    batch,
    seq_len, 
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles one block of the output tensor
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate output dimensions
    m_offset = batch_idx * seq_len + seq_idx
    n_offset = tl.program_id(2) * BLOCK_SIZE_N
    
    # Boundary checks
    m_end = batch * seq_len
    n_end = out_features
    
    # Load scales for this column block
    scale_ptrs = scale_ptr + n_offset
    scale_block = tl.load(scale_ptrs + tl.arange(0, min(BLOCK_SIZE_N, n_end - n_offset)))
    
    # Compute linear transformation for this position
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Process input features in blocks
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, in_features)
        
        # Load input data
        x_ptrs = x_ptr + (batch_idx * seq_len * in_features + seq_idx * in_features + k)
        x_block = tl.load(x_ptrs + tl.arange(0, min(BLOCK_SIZE_K, k_end - k)), mask=None)
        
        # Load weights for this k block
        weight_ptrs = weight_ptr + (n_offset * in_features + k)
        weight_block = tl.load(weight_ptrs + tl.arange(0, BLOCK_SIZE_N * min(BLOCK_SIZE_K, k_end - k)), 
                             stride=in_features, mask=None)
        weight_block = tl.reshape(weight_block, [BLOCK_SIZE_N, min(BLOCK_SIZE_K, k_end - k)])
        
        # Matrix multiplication
        acc += tl.dot(x_block, weight_block, acc_type=tl.float32)
    
    # Apply element-wise multiplication with scales
    result = acc * scale_block
    
    # Store result
    output_ptrs = out_ptr + (m_offset * out_features + n_offset)
    tl.store(output_ptrs + tl.arange(0, min(BLOCK_SIZE_N, n_end - n_offset)), result)

@torch.fx.wrap
def triton_linear_mul_fused_mmpose(in_3, in_0, in_1, in_2):
    # Get tensor shapes and properties
    batch, seq_len, in_features = in_3.shape
    out_features = in_0.shape[0]  # in_0 is [out_features, in_features]
    
    # Create output tensors
    linear_out = torch.empty([batch, seq_len, out_features], dtype=torch.float32, device=in_3.device)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_M = 1  # Process one batch*seq position at a time
    BLOCK_SIZE_N = 256  # Output features per block
    BLOCK_SIZE_K = 64   # Input features per block
    
    # Calculate grid dimensions
    num_batch_seq = batch * seq_len
    num_out_blocks = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Compute linear transformation: tmp_2 = linear(in_3, in_0, None)
    # Create a simple scale tensor without using forbidden APIs
    scale_data = torch.ones(1, device=in_3.device, dtype=torch.float32)
    
    # We'll use the existing kernel but ignore the scale parameter (pass None)
    linear_mul_kernel_fused[(num_batch_seq, num_out_blocks)](
        in_3, in_0, scale_data, linear_out,
        batch, seq_len, in_features, out_features,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Compute broadcasted multiplication: tmp_3 = in_2 * in_1
    # This is efficient on GPU due to built-in broadcasting
    mul_out = in_2 * in_1.reshape(1, 1, -1)  # Broadcast [256] to [1, 1, 256]
    
    return mul_out, linear_out  # Return (tmp_3, tmp_2)

# Pattern for MMPose: tmp_2 = linear(in_3, tmp_0, None), tmp_3 = in_2 * tmp_1
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_0, None)
    tmp_3 = in_2 * tmp_1
    return tmp_3, tmp_2

# Extract arguments for the replacement
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_3, in_0, in_1, in_2)  # Pass inputs to triton function in correct order

# Return the optimized function
def replacement_func():
    return triton_linear_mul_fused_mmpose