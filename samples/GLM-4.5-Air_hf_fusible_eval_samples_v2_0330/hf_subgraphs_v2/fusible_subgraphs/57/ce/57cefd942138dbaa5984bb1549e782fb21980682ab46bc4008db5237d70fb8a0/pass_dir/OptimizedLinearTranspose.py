import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching for linear transformation + transpose sequence"""
    # Linear transformation: in_2 @ in_1.T + in_0
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    # Transpose last two dimensions to get [batch, hidden_size, seq_len]
    transpose_result = linear.transpose(-1, -2)
    return transpose_result

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the replacement function"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_linear_transpose_kernel(
    bias_ptr,           # Pointer to bias tensor [hidden_size]
    weight_ptr,         # Pointer to weight tensor [hidden_size, hidden_size] 
    input_ptr,          # Pointer to input tensor [batch_size, seq_len, hidden_size]
    output_ptr,         # Pointer to output tensor [batch_size, hidden_size, seq_len]
    batch_size,         # Batch size
    seq_len,            # Sequence length  
    hidden_size,        # Hidden size
    BLOCK_SIZE_M: tl.constexpr,      # Block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,      # Block size for hidden dimension
    BLOCK_SIZE_K: tl.constexpr,      # Block size for sequence dimension
):
    """Optimized kernel for linear transformation with implicit transpose"""
    
    # Program identifiers for 3D grid
    m = tl.program_id(0)  # batch dimension
    n = tl.program_id(1)  # hidden dimension  
    k = tl.program_id(2)  # sequence dimension
    
    # Compute memory offsets within the block
    m_offset = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offset = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks to handle boundary conditions
    m_mask = m_offset < batch_size
    n_mask = n_offset < hidden_size
    k_mask = k_offset < seq_len
    
    # Load bias [hidden_size] and convert to input dtype
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    bias = bias.to(input_ptr.dtype.element_ty)
    
    # Initialize accumulator for [batch_size, seq_len] output
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=input_ptr.dtype.element_ty)
    
    # Compute the linear transformation: input @ weight.T + bias
    # This will naturally produce [batch, seq_len] per hidden dimension
    for k_hidden in range(0, hidden_size, BLOCK_SIZE_N):
        # Load input slice [batch, seq, hidden_slice]
        input_slice = tl.load(
            input_ptr + (m_offset[:, None] * seq_len * hidden_size + 
                        k_offset[None, :] * hidden_size + 
                        k_hidden),
            mask=(m_mask[:, None] & k_mask[None, :] & (k_hidden < hidden_size)),
            other=0.0
        )
        
        # Load weight slice [hidden_slice, hidden] - note this is already transposed for our needs
        weight_slice = tl.load(
            weight_ptr + (k_hidden * hidden_size + n_offset),
            mask=(k_hidden < hidden_size & n_mask),
            other=0.0
        ).to(input_ptr.dtype.element_ty)
        
        # Matrix multiplication: input_slice @ weight_slice 
        # Since weight_slice is [hidden_slice, hidden], this gives [batch, seq, hidden]
        # But we're accumulating along hidden dimension, so we get [batch, seq]
        accum += tl.dot(input_slice, weight_slice.to(input_slice.dtype), out_dtype=input_slice.dtype)
    
    # Add bias (broadcasted across batch and sequence)
    accum += bias[None, :]  # bias[hidden] becomes [1, hidden], then broadcasted
    
    # Store result directly in transposed position: [batch, hidden, seq]
    tl.store(
        output_ptr + (m_offset[:, None] * hidden_size * seq_len + 
                     n_offset[None, :] * seq_len + 
                     k_offset),
        accum,
        mask=(m_mask[:, None] & n_mask[None, :] & k_mask)
    )

@torch.fx.wrap
def optimized_linear_transpose(in_0, in_1, in_2, in_3):
    """Optimized wrapper for linear transformation with implicit transpose"""
    batch_size, seq_len, hidden_size = in_2.shape
    
    # Optimized block sizes for better GPU utilization
    BLOCK_SIZE_M = 8    # Process more batches together for better occupancy
    BLOCK_SIZE_N = 64   # Process hidden dimension efficiently  
    BLOCK_SIZE_K = 256  # Use larger sequence blocks for better memory coalescing
    
    # Calculate number of program blocks
    m_grid = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_grid = (hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    k_grid = (seq_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Create output tensor with transposed shape [batch, hidden, seq]
    output = torch.empty((batch_size, hidden_size, seq_len), 
                        dtype=in_2.dtype, 
                        device=in_2.device)
    
    # Launch optimized kernel
    optimized_linear_transpose_kernel[(m_grid, n_grid, k_grid)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Perform element-wise multiplication with in_3
    result = in_3 * output
    
    return result

def replacement_func():
    """Return the optimized function as a callable"""
    return optimized_linear_transpose