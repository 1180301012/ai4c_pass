import torch
import triton
import triton.language as tl

def pattern(k_tensor):
    tmp_13 = k_tensor.transpose(-2, -1)
    return tmp_13

def replacement_args(k_tensor):
    return (k_tensor,)

@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    batch_size, num_heads, seq_len, features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate work per program
    total_elements = batch_size * num_heads * seq_len * features
    elements_per_program = (total_elements + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    if start_idx >= end_idx:
        return
    
    # Process elements in blocks
    for idx in range(start_idx, end_idx):
        # Calculate coordinates
        batch_idx = idx // (num_heads * seq_len * features)
        head_idx = (idx // (seq_len * features)) % num_heads
        seq_idx = (idx // features) % seq_len
        feat_idx = idx % features
        
        # Calculate transposed coordinates
        trans_idx = (batch_idx * num_heads + head_idx) * seq_len * features + feat_idx * seq_len + seq_idx
        
        # Load and store
        val = tl.load(input_ptr + idx, other=0.0)
        tl.store(output_ptr + trans_idx, val)

@torch.fx.wrap
def optimized_k_transpose(k_tensor):
    batch_size, seq_len, num_heads, features = k_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(k_tensor)
    
    # Launch kernel with optimized block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    
    # Calculate grid dimensions - number of programs needed
    total_elements = k_tensor.numel()
    programs_needed = (total_elements + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    # Limit the number of programs to avoid oversubscription
    max_programs = 1024  # Adjust based on GPU capabilities
    if programs_needed > max_programs:
        programs_needed = max_programs
    
    kernels_per_program = (programs_needed + max_programs - 1) // max_programs
    
    for i in range(kernels_per_program):
        start_program = i * max_programs
        end_program = min((i + 1) * max_programs, programs_needed)
        
        transpose_kernel[(end_program - start_program,)](
            input_ptr=k_tensor,
            output_ptr=output,
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            features=features,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    
    return output

def replacement_func():
    return optimized_k_transpose