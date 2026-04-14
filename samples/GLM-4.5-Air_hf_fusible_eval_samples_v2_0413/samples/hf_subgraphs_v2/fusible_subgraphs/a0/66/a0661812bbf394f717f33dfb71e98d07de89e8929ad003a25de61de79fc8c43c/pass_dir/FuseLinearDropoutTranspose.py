import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the complete computation pattern that produces both return values
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear  # This represents the dropout output, but we'll do dropout in kernel
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4

def replacement_args(in_0, in_1, in_2):
    # For fusion, we need the dropout probability. This will be provided by the match
    # The actual dropout probability is captured during pattern matching
    return (in_0, in_1, in_2)

@triton.jit
def fused_linear_dropout_transpose_kernel(
    bias_ptr,          # bias [D]
    weight_ptr,        # weight [D, H] 
    input_ptr,         # input [B, S, H]
    output_ptr,        # output [B, S, D]
    transposed_ptr,    # transposed [B, D, S]
    B: tl.constexpr,    # batch_size = 1
    S: tl.constexpr,    # sequence length
    H: tl.constexpr,    # hidden_dim  
    D: tl.constexpr,    # output_dim
    dropout_p: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program identifiers for 2D grid
    pid_m = tl.program_id(0)  # along sequence dimension
    pid_n = tl.program_id(1)  # along output dimension
    
    # Compute offsets within the blocks
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    m_mask = m_offset < S
    n_mask = n_offset < D
    
    # Load bias vector [D]
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    
    # Accumulator for linear transformation [BLOCK_SIZE_M, BLOCK_SIZE_N]
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    
    # Loop over hidden dimension
    for k in range(0, H, BLOCK_SIZE_M):
        k_block = tl.arange(0, BLOCK_SIZE_M) + k
        k_mask = k_block < H
        
        # Load input block [BLOCK_SIZE_M, H] 
        input_block = tl.load(input_ptr + m_offset[:, None] * H + k_block[None, :], 
                             mask=k_mask[None, :], other=0.0)
        
        # Load weight block [H, BLOCK_SIZE_N]
        weight_block = tl.load(weight_ptr + k_block[:, None] * D + n_offset[None, :], 
                              mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Matrix multiplication: accumulate += input_block @ weight_block
        accumulator += tl.dot(input_block, weight_block, out_dtype=tl.float32)
    
    # Add bias [BLOCK_SIZE_M, BLOCK_SIZE_N]
    accumulator = accumulator + bias[None, :]
    
    # Convert back to float16 (original dtype)
    output_block = accumulator.to(tl.float16)
    
    # Apply dropout if probability > 0
    if dropout_p > 0.0:
        # Generate random numbers for dropout using threadIdx
        random_seed = tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
        mask = tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
        random_base = (pid_m * BLOCK_SIZE_M + mask // BLOCK_SIZE_N) * 1000 + (pid_n * BLOCK_SIZE_N + mask % BLOCK_SIZE_N)
        random_numbers = (random_base * 1664525 + 1013904223) % (2**32) / (2**32)  # Simple LCG
        dropout_mask = random_numbers > dropout_p
        output_block = output_block * dropout_mask.to(tl.float16)
        # Scale to maintain expected value
        output_block = output_block * (1.0 / (1.0 - dropout_p))
    
    # Store output [B, S, D] - assuming B=1, so stride 1*S + offset
    output_base = pid_m * D + pid_n * S * D
    output_offsets = m_offset[:, None] * D + n_offset[None, :]
    tl.store(output_ptr + output_base + output_offsets, output_block, mask=m_mask[:, None] & n_mask[None, :])
    
    # Store transposed output [B, D, S] - assuming B=1, so stride 1*D + offset  
    transposed_base = pid_n * S + pid_m * D * S
    transposed_offsets = n_offset[:, None] * S + m_offset[None, :]
    tl.store(transposed_ptr + transposed_base + transposed_offsets, output_block.T, mask=n_mask[:, None] & m_mask[None, :])

@torch.fx.wrap
def fused_linear_dropout_transpose(in_0, in_1, in_2, dropout_p):
    # Set up dimensions from input tensors
    B = in_2.size(0)
    S = in_2.size(1) 
    H = in_2.size(2)
    D = in_0.size(0)  # bias dimension
    
    # Choose optimal block sizes based on problem dimensions
    # For large matrices (like S=249, D=768), use smaller blocks for better occupancy
    if S > 128 and D > 128:
        BLOCK_SIZE_M = 32  # sequence dimension
        BLOCK_SIZE_N = 64  # output dimension  
    else:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
    
    # Create output tensors
    output = torch.empty((B, S, D), device=in_2.device, dtype=in_2.dtype)
    transposed = torch.empty((B, D, S), device=in_2.device, dtype=in_2.dtype)
    
    # Calculate grid dimensions
    num_blocks_m = (S + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (D + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_linear_dropout_transpose_kernel[(num_blocks_m, num_blocks_n)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        transposed_ptr=transposed,
        B=B,
        S=S,
        H=H,
        D=D,
        dropout_p=float(dropout_p),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output, transposed

def replacement_func():
    return fused_linear_dropout_transpose