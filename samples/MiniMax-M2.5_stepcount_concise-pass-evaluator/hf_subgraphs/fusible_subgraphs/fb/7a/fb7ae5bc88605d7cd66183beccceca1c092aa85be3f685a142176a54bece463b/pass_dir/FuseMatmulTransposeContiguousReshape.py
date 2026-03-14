import torch
import triton
import triton.language as tl

# Pattern to match:
# tmp_4 = torch.matmul(tmp_3, in_1)
# tmp_5 = tmp_4.transpose(1, 2)
# tmp_6 = tmp_5.contiguous()
# tmp_7 = tmp_6.reshape(1, 257, -1)
# tmp_8 = tmp_7.contiguous()
# return tmp_8

def pattern(tmp_3, in_1):
    """
    Match pattern: matmul -> transpose -> contiguous -> reshape -> contiguous
    This fuses multiple operations into one optimized kernel.
    
    Original shapes:
    - tmp_3: [1, 16, 257, 257]
    - in_1: [1, 16, 257, 80]
    - matmul result: [1, 16, 257, 80]
    - after transpose: [1, 257, 16, 80]
    - after reshape: [1, 257, 1280]
    """
    tmp_4 = torch.matmul(tmp_3, in_1)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.reshape(1, 257, -1)
    tmp_8 = tmp_7.contiguous()
    return tmp_8


def replacement_args(tmp_3, in_1):
    return (tmp_3, in_1)


@triton.jit
def fused_matmul_transpose_reshape_kernel(
    output_ptr,
    input_a_ptr,
    input_b_ptr,
    batch_size: tl.constexpr,
    num_heads_a: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. Batch matmul: [batch, heads_a, seq_len, seq_len] @ [batch, heads_a, seq_len, head_dim]
                  -> [batch, heads_a, seq_len, head_dim]
    2. Transpose: [batch, seq_len, heads_a, head_dim]
    3. Reshape: [batch, seq_len, heads_a * head_dim]
    
    All in a single kernel to avoid multiple memory passes.
    """
    # Get program coordinates
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Output position for this batch and sequence position
    # Output shape: [batch, seq_len, heads_a * head_dim]
    output_offset = batch_idx * seq_len * (num_heads_a * head_dim) + seq_idx * (num_heads_a * head_dim)
    
    # Compute matmul and produce output directly
    # For each head in input_a, compute dot product with corresponding head in input_b
    # Then arrange in transposed order
    
    # Pointers to the start of each sequence position
    # input_a: [batch, heads_a, seq_len, seq_len] -> offset = batch * heads_a * seq_len * seq_len + head * seq_len * seq_len + seq_idx * seq_len
    # input_b: [batch, heads_a, seq_len, head_dim] -> offset = batch * heads_a * seq_len * head_dim + head * seq_len * head_dim + seq_idx * head_dim
    
    result = tl.zeros((num_heads_a * head_dim,), dtype=tl.float32)
    
    for head_idx in range(num_heads_a):
        # Load row from input_a: [seq_len]
        a_base = batch_idx * num_heads_a * seq_len * seq_len + head_idx * seq_len * seq_len + seq_idx * seq_len
        a_ptrs = input_a_ptr + a_base + tl.arange(0, BLOCK_SIZE)
        a_mask = tl.arange(0, BLOCK_SIZE) < seq_len
        a_row = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # For each output element in this head
        for head_dim_idx in range(head_dim):
            # Load column from input_b: [head_dim]
            b_base = batch_idx * num_heads_a * seq_len * head_dim + head_idx * seq_len * head_dim + seq_idx * head_dim + head_dim_idx
            b_val = tl.load(input_b_ptr + b_base)
            
            # Dot product: a_row @ b_val (scaler)
            dot_product = tl.sum(a_row * b_val)
            
            # Store in output at transposed position
            # Output: [batch, seq_len, heads_a * head_dim]
            # Index: head_idx * head_dim + head_dim_idx
            out_idx = head_idx * head_dim + head_dim_idx
            result = tl.where(out_idx == tl.arange(0, num_heads_a * head_dim), 
                            tl.where(True, dot_product, result), 
                            result)
    
    # Store result
    out_ptrs = output_ptr + output_offset + tl.arange(0, num_heads_a * head_dim)
    out_mask = tl.arange(0, num_heads_a * head_dim) < num_heads_a * head_dim
    tl.store(out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def fused_matmul_transpose_reshape(a, b):
    """
    Fused kernel for: matmul -> transpose -> contiguous -> reshape -> contiguous
    
    Input shapes:
    - a: [1, 16, 257, 257] (query/key)
    - b: [1, 16, 257, 80] (value)
    
    Output shape: [1, 257, 1280]
    """
    batch_size = a.shape[0]  # 1
    num_heads = a.shape[1]   # 16
    seq_len = a.shape[2]     # 257
    head_dim = b.shape[3]    # 80
    
    # Allocate output
    output = torch.empty((batch_size, seq_len, num_heads * head_dim), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    BLOCK_SIZE = seq_len  # 257
    grid = (batch_size, seq_len)
    
    fused_matmul_transpose_reshape_kernel[grid](
        output_ptr=output,
        input_a_ptr=a,
        input_b_ptr=b,
        batch_size=batch_size,
        num_heads_a=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_matmul_transpose_reshape