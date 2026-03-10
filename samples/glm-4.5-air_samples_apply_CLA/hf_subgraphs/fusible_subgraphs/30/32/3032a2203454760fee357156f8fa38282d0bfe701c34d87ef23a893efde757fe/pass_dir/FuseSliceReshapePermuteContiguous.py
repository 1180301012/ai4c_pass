import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    tmp_3 = tmp_2[..., 1:, :]
    tmp_4 = tmp_3.reshape(1, 12, 12, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(tmp_2):
    return (tmp_2,)

# Optimized kernel for slice + reshape + permute + contiguous
@triton.jit
def fused_kernel_slice_reshape_permute(
    input_ptr, 
    output_ptr,
    n_seq: tl.constexpr,  # 144 (145 - 1)
    hidden_size: tl.constexpr,  # 512
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles a [BLOCK_M, BLOCK_N] tile
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = m < n_seq
    
    # Load data directly from the sliced region (skip index 0)
    offset = (m + 1) * hidden_size + n
    data = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Store in permuted and reshaped layout: [1, hidden_size, 12, 12]
    # We treat it as [hidden_size, n_seq]
    output_offset = n * n_seq + m
    tl.store(output_ptr + output_offset, data, mask=mask)

@torch.fx.wrap
def kernel_fused_slice_reshape_permute(input_tensor):
    # Input shape: [1, 145, 512]
    n_seq = 144  # 145 - 1
    hidden_size = 512
    
    # Calculate output size: [1, hidden_size, 12, 12] -> [hidden_size, 12, 12]
    output_size = (hidden_size, 12, 12)
    output = torch.zeros(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Reshape input to [1, 145, 512] for easier handling
    input_reshaped = input_tensor.reshape(-1, hidden_size)
    
    # Launch kernel with appropriate grid
    BLOCK_M = 64
    BLOCK_N = 8
    num_m = (n_seq + BLOCK_M - 1) // BLOCK_M
    num_n = (hidden_size + BLOCK_N - 1) // BLOCK_N
    
    fused_kernel_slice_reshape_permute[(num_m, num_n, 1)](
        input_reshaped,
        output.reshape(hidden_size, -1),  # Reshape to [hidden_size, 144]
        n_seq,
        hidden_size,
        BLOCK_M,
        BLOCK_N,
    )
    
    return output

def replacement_func():
    return kernel_fused_slice_reshape_permute