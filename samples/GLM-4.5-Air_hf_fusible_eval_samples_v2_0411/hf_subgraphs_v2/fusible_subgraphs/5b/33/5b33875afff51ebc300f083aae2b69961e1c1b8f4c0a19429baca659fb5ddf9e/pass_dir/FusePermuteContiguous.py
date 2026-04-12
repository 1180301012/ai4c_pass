import torch
import triton
import triton.language as tl

def pattern(matmul_result):
    tmp_5 = matmul_result.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(matmul_result):
    return (matmul_result,)

@triton.jit
def permute_contiguous_kernel(
    input_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Decompose pid into m (seq) and n (head_dim)
    num_m_blocks = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_blocks = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    m = pid // num_n_blocks  # seq_len block index
    n = pid % num_n_blocks   # head_dim block index
    
    # Thread indices within the block
    offsets_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # seq_len
    offsets_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # head_dim
    
    # Mask for valid range
    mask_m = offsets_m < seq_len
    mask_n = offsets_n < head_dim
    
    # Process all heads in this program
    for head_idx in range(num_heads):
        # Input: [batch, heads, seq_len, head_dim]
        # Calculate input offset: b * (H * S * D) + h * (S * D) + s * D + d
        # For batch=1: h * (S * D) + s * D + d
        input_offset = head_idx * (seq_len * head_dim) + offsets_m[:, None] * head_dim + offsets_n[None, :]
        
        # Load input data
        input_data = tl.load(
            input_ptr + input_offset,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Output: [batch, seq_len, heads, head_dim]  
        # Calculate output offset: b * (S * H * D) + s * (H * D) + h * D + d
        # For batch=1: s * (H * D) + h * D + d
        output_offset = offsets_m[:, None] * (num_heads * head_dim) + head_idx * head_dim + offsets_n[None, :]
        
        # Store output data
        tl.store(
            output_ptr + output_offset,
            input_data,
            mask=mask_m[:, None] & mask_n[None, :]
        )

@torch.fx.wrap
def permute_contiguous_wrapper(matmul_result):
    batch_size, num_heads, seq_len, head_dim = matmul_result.shape
    
    BLOCK_SIZE_M = 64  # seq_len dim
    BLOCK_SIZE_N = 32  # head_dim dim
    
    # Calculate grid size
    num_m_blocks = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_blocks = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    total_blocks = num_m_blocks * num_n_blocks
    
    # Create output tensor with correct permuted shape
    output_shape = (batch_size, seq_len, num_heads, head_dim)
    output = torch.empty(output_shape, dtype=matmul_result.dtype, device=matmul_result.device)
    
    # Launch kernel to perform the permutation
    permute_contiguous_kernel[(total_blocks,)](
        matmul_result, output,
        batch_size, num_heads, seq_len, head_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return permute_contiguous_wrapper