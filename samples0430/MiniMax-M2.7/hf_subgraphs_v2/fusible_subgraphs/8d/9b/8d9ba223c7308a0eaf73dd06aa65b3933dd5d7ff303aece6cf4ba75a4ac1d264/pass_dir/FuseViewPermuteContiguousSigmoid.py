import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match pattern: view -> permute -> contiguous -> sigmoid
    
    This pattern matches the sequence:
    tmp_6 = x.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    
    Returns: tmp_9
    
    Input x is (4096, num_heads) from advanced indexing
    """
    tmp_6 = x.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    return tmp_9


def replacement_args(x):
    return (x,)


@triton.jit
def fused_reshape_permute_sigmoid_kernel(
    input_ptr,
    output_ptr,
    total_elements: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, num_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: reshape + permute + sigmoid
    
    Input: (total_elements, num_heads) = (4096, num_heads) in row-major
    Output: (num_heads, M, N) - permuted and sigmoid applied
    
    The kernel handles:
    1. Reshape input (total_elements, num_heads) to (M, N, num_heads)
    2. Permute to (num_heads, M, N)
    3. Apply sigmoid
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Output index: (head, i, j) -> flat offset
    head = offsets // (M * N)
    remainder = offsets % (M * N)
    i = remainder // N
    j = remainder % N
    
    # Input: (total_elements, num_heads), we need element at position 'idx'
    # For output (head, i, j), we need input[element_idx, head]
    # where element_idx = i * N + j = remainder
    element_idx = remainder
    input_idx = element_idx * num_heads + head
    
    # Load value from input (bf16/fp16/fp32)
    x_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Convert to fp32 for exp computation (exp doesn't support bf16/fp16)
    x_fp32 = x_val.to(tl.float32)
    
    # Compute sigmoid in fp32
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x_fp32))
    
    # Convert back to original dtype
    sigmoid_out = sigmoid_x.to(x_val.dtype)
    
    # Store to output (head, i, j) = (num_heads, M, N)
    output_idx = head * M * N + i * N + j
    
    tl.store(output_ptr + output_idx, sigmoid_out, mask=mask)


@torch.fx.wrap
def fused_reshape_permute_sigmoid(x):
    """
    Fused kernel: view + permute + contiguous + sigmoid
    
    Input: x with shape (4096, num_heads)
    Output: (num_heads, 64, 64) with sigmoid applied
    """
    num_heads = x.shape[1]
    M, N = 64, 64
    total_elements = M * N
    
    output = torch.empty((num_heads, M, N), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_reshape_permute_sigmoid_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        total_elements=total_elements,
        M=M, N=N, num_heads=num_heads,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_reshape_permute_sigmoid