import torch
import triton
import triton.language as tl

# Pattern matching for convit_small computation
def pattern(in_0, in_1):
    """Match linear + reshape + permute + split + transpose sequence for convit_small"""
    # Linear operation
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    
    # Reshape to 5D tensor - convit_small specific
    tmp_2 = tmp_1.reshape(1, 197, 3, 9, 48)
    
    # Permute dimensions: (2, 0, 3, 1, 4)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    
    # Split along dim 0 (unbind)
    tmp_4 = tmp_3.unbind(0)
    
    # Extract the three components
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    
    # Transpose the middle component
    tmp_8 = tmp_6.transpose(-2, -1)
    
    return tmp_5, tmp_8, tmp_7

def replacement_args(in_0, in_1):
    """Extract arguments for the fused kernel"""
    return in_0, in_1

@triton.jit
def optimized_linear_kernel(
    weight_ptr,           # [D_out, D_in] weight matrix
    input_ptr,            # [batch, seq_len, D_in] input tensor  
    output_ptr,           # [batch, seq_len, D_out] output tensor
    batch,                # batch size (1)
    seq_len,              # sequence length (197)
    D_in,                 # input dimension (432/192)
    D_out,                # output dimension (1296/576)
    BLOCK_SIZE_M: tl.constexpr,  # block size for seq_len
    BLOCK_SIZE_N: tl.constexpr,  # block size for D_in/D_out
):
    """Optimized linear kernel using Triton"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Get tile bounds  
    m_mask = pid_m < (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_mask = pid_n < (D_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    if not (m_mask and n_mask):
        return
    
    # Block offsets
    m_block_start = pid_m * BLOCK_SIZE_M
    n_block_start = pid_n * BLOCK_SIZE_N
    
    # Tile dimensions
    m = m_block_start + tl.arange(0, BLOCK_SIZE_M)  # seq_len positions
    n = n_block_start + tl.arange(0, BLOCK_SIZE_N)  # D_out positions
    
    # Masks
    m_mask = m < seq_len
    n_mask = n < D_out
    
    # Accumulators for each output position
    accums = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Vectorized matmul - process multiple D_in elements per thread
    k_mask = tl.arange(0, BLOCK_SIZE_N) < D_in
    for i in range(0, D_in, BLOCK_SIZE_N):
        k = i + tl.arange(0, BLOCK_SIZE_N)
        k_mask = k < D_in
        
        # Load input slice for current batch and sequence positions
        input_slice = tl.load(input_ptr + m[:, None] * D_in + k[None, :], 
                             mask=m_mask[:, None] & k_mask[None, :], 
                             other=0.0)
        
        # Load weights for current D_out positions
        weights = tl.load(weight_ptr + n[:, None] * D_in + k[None, :], 
                         mask=n_mask[:, None] & k_mask[None, :], 
                         other=0.0)
        
        # Accumulate results
        accums += tl.sum(input_slice * weights, dim=1)
    
    # Store results
    tl.store(output_ptr + m[:, None] * D_out + n[None, :], 
             accums[:, None], 
             mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def optimized_attn_forward(weight, input_tensor):
    """Forward pass using optimized linear kernel + tensor reshaping"""
    batch, seq_len, D_in = input_tensor.shape
    D_out = weight.shape[0]
    
    # Optimized linear operation using Triton kernel
    linear_output = torch.empty(batch, seq_len, D_out, dtype=torch.float32, device=input_tensor.device)
    
    # Launch optimized linear kernel
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (D_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_linear_kernel[(
        grid_m, 
        grid_n
    )](
        weight, 
        input_tensor, 
        linear_output,
        batch, 
        seq_len, 
        D_in, 
        D_out,
        BLOCK_SIZE_M, 
        BLOCK_SIZE_N
    )
    
    # Now continue with the tensor operations for reshaping and splitting
    # Reshape to 5D: [1, 197, 1296] -> [1, 197, 3, 9, 48] 
    reshaped = linear_output.reshape(1, seq_len, 3, 9, 48)
    
    # Permute: (2, 0, 3, 1, 4) -> [3, 1, 9, 197, 48]
    permuted_reshaped = reshaped.permute(2, 0, 3, 1, 4)
    
    # Unbind along dim 0 and process
    outputs = permuted_reshaped.unbind(0)
    
    # Final output tensors
    # tmp_5: first tensor [1, 9, 197, 48]
    output1 = outputs[0]
    
    # tmp_8: second tensor [1, 9, 197, 48] -> transpose last 2 dims -> [1, 197, 9, 48]
    output2 = outputs[1].transpose(-2, -1)
    
    # tmp_7: third tensor [1, 9, 197, 48]  
    output3 = outputs[2]
    
    return output1, output2, output3

def replacement_func():
    return optimized_attn_forward