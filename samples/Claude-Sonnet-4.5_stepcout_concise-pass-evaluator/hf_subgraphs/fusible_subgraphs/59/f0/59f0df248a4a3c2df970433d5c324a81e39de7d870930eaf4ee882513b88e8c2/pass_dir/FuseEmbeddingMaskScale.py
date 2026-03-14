import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight):
    """
    Pattern: Embedding lookup + masking padding tokens
    """
    tmp_3 = torch.nn.functional.embedding(input_ids, embedding_weight, 1, None, 2.0, False, False)
    tmp_4 = input_ids.__eq__(2)
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_3.masked_fill(tmp_5, 0.0)
    return tmp_6

def replacement_args(input_ids, embedding_weight):
    return (input_ids, embedding_weight)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_embedding_mask_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    M,  # batch_size * seq_len
    N,  # embedding_dim
    stride_input_ids,
    stride_emb_row,
    stride_emb_col,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load input_ids for this block
    input_ids_ptrs = input_ids_ptr + offs_m * stride_input_ids
    input_ids = tl.load(input_ids_ptrs, mask=mask_m, other=0)
    
    # Check if padding token (id == 2)
    is_padding = (input_ids == 2)
    
    # Load embeddings and apply masking
    # Broadcast is_padding to match embedding dimensions
    is_padding_expanded = is_padding[:, None]
    
    # Process in chunks - load embeddings for each token in the block
    output_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for i in range(BLOCK_SIZE_M):
        if offs_m[i] < M:
            token_id = tl.load(input_ids_ptr + offs_m[i] * stride_input_ids)
            
            # Load embedding row for this token
            emb_ptrs = embedding_weight_ptr + token_id * stride_emb_row + offs_n * stride_emb_col
            emb_row = tl.load(emb_ptrs, mask=mask_n, other=0.0)
            
            # Apply masking: if token_id == 2, set to 0
            is_pad = (token_id == 2)
            if is_pad:
                emb_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            
            # Store in output block
            for j in range(BLOCK_SIZE_N):
                if offs_n[j] < N:
                    output_block[i, j] = emb_row[j]
    
    # Store output
    output_ptrs = output_ptr + (offs_m[:, None] * stride_out_row + offs_n[None, :] * stride_out_col)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(output_ptrs, output_block, mask=mask)

@torch.fx.wrap
def fused_embedding_mask(input_ids, embedding_weight):
    # Get shapes
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    embedding_dim = embedding_weight.shape[1]
    
    M = batch_size * seq_len
    N = embedding_dim
    
    # Flatten input_ids
    input_ids_flat = input_ids.contiguous().view(-1)
    
    # Output tensor
    output = torch.empty((batch_size, seq_len, embedding_dim), 
                         dtype=embedding_weight.dtype, 
                         device=embedding_weight.device)
    
    # Grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    fused_embedding_mask_kernel[grid](
        input_ids_flat,
        embedding_weight,
        output.view(M, N),
        M, N,
        input_ids_flat.stride(0),
        embedding_weight.stride(0),
        embedding_weight.stride(1),
        output.view(M, N).stride(0),
        output.view(M, N).stride(1),
    )
    
    return output

def replacement_func():
    return fused_embedding_mask