import torch
import triton
import triton.language as tl

@triton.jit
def linear_mul_kernel_fused_bf16(
    x_ptr,           # Input tensor for linear: [batch, seq_len, in_features]
    weight_ptr,      # Weight matrix: [out_features, in_features]
    mul_input_ptr,   # Input for element-wise multiplication: [batch, seq_len, out_features]
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
    
    # Load multiplication input for this position
    mul_ptrs = mul_input_ptr + (m_offset * out_features + n_offset)
    mul_block = tl.load(mul_ptrs + tl.arange(0, min(BLOCK_SIZE_N, n_end - n_offset)))
    
    # Apply element-wise multiplication
    result = acc * mul_block
    
    # Store result (as bfloat16 to match input dtype)
    output_ptrs = out_ptr + (m_offset * out_features + n_offset)
    tl.store(output_ptrs + tl.arange(0, min(BLOCK_SIZE_N, n_end - n_offset)), result.to(tl.float16))

def triton_linear_mul_fused_smollm3(x, weight, mul_input):
    """
    Optimized fused linear + element-wise multiplication for SmolLM3 pattern
    """
    # Get tensor shapes and properties
    batch, seq_len, in_features = x.shape
    out_features = weight.shape[0]  # weight is [out_features, in_features]
    
    # Create output tensor with bfloat16 dtype (use tuple, not list)
    out = torch.empty((batch, seq_len, out_features), dtype=torch.bfloat16, device=x.device)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_M = 1  # Process one batch*seq position at a time
    BLOCK_SIZE_N = 512  # Output features per block
    BLOCK_SIZE_K = 128   # Input features per block
    
    # Calculate grid dimensions
    num_batch_seq = batch * seq_len
    num_out_blocks = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    linear_mul_kernel_fused_bf16[(num_batch_seq, num_out_blocks)](
        x,
        weight,
        mul_input,
        out,
        batch,
        seq_len,
        in_features,
        out_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return out

# Pattern for SmolLM3: tmp_1 = linear(in_1, tmp_0, None), tmp_2 = in_2 * tmp_1
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_2 = in_2 * tmp_1
    return tmp_2,

# Extract arguments for the replacement
def replacement_args(in_0, in_1, in_2):
    return (in_1, in_0, in_2)  # Pass inputs to triton function in correct order

# Return the optimized function
def replacement_func():
    return triton_linear_mul_fused_smollm3