import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_3):
    tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_4 = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    return tmp_5

# Argument extraction function
def replacement_args(in_3):
    return (in_3,)

# Triton kernel
@triton.jit
def rotate_kernel(
    input_ptr,
    output_ptr,
    n_heads,
    seq_len,
    head_dim,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
):
    pid_heads = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_head_dim = tl.program_id(2)
    
    offs_head_dim = pid_head_dim * BLOCK_HEAD_DIM + tl.arange(0, BLOCK_HEAD_DIM)
    offs_head = pid_heads * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    offs_seq = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    
    mask_head_dim = offs_head_dim < head_dim
    mask_head = offs_head < n_heads
    mask_seq = offs_seq < seq_len
    
    input_indices = (offs_head[:, None, None] * seq_len * head_dim + 
                     offs_seq[None, :, None] * head_dim + 
                     offs_head_dim[None, None, :])
    input_data = tl.load(input_ptr + input_indices, mask=mask_head[:, None, None] & mask_seq[None, :, None] & mask_head_dim[None, None, :], other=0.0)
    
    is_odd = (offs_head_dim & 1) == 1
    output_data = tl.where(is_odd, -input_data, input_data)
    
    output_indices = input_indices
    tl.store(output_ptr + output_indices, output_data, mask=mask_head[:, None, None] & mask_seq[None, :, None] & mask_head_dim[None, None, :])

# Kernel wrapper
@torch.fx.wrap
def rotate(input):
    _, n_heads, seq_len, head_dim = input.shape
    BLOCK_HEADS = 8
    BLOCK_SEQ = 32
    BLOCK_HEAD_DIM = 32
    
    num_heads_blocks = (n_heads + BLOCK_HEADS - 1) // BLOCK_HEADS
    num_seq_blocks = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    num_head_dim_blocks = (head_dim + BLOCK_HEAD_DIM - 1) // BLOCK_HEAD_DIM
    
    output = torch.empty_like(input)
    
    rotate_kernel[(num_heads_blocks, num_seq_blocks, num_head_dim_blocks)](
        input_ptr=input,
        output_ptr=output,
        n_heads=n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_HEADS=BLOCK_HEADS,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM
    )
    
    return output

# Replacement function
def replacement_func():
    return rotate