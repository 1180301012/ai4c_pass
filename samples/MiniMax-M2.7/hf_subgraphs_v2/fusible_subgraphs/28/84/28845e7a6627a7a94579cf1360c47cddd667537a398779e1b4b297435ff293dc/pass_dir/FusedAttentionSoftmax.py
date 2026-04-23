import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_softmax_kernel(
    mask_ptr,
    attn_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    n_rows: tl.constexpr,
    row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused attention softmax kernel that:
    1. Adds mask to attention scores
    2. Applies softmax along the last dimension
    
    mask: [1, 1, N, N] 
    attn: [1, H, N, N]
    output: [1, H, N, N]
    """
    # Get row index
    row_idx = tl.program_id(0)
    head_idx = row_idx // n_elements
    row_in_head = row_idx % n_elements
    
    # Each thread block handles a row, processing all columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask_ptrs = mask_ptr + col_offsets  # Mask is broadcasted
    attn_ptrs = attn_ptr + row_in_head * row_stride + col_offsets
    
    mask = tl.load(mask_ptrs, mask=col_offsets < n_elements, other=0.0)
    attn = tl.load(attn_ptrs, mask=col_offsets < n_elements, other=0.0)
    
    # Add mask to attention (mask contains -inf for masked positions)
    val = attn + mask
    
    # Compute softmax: subtract max for numerical stability
    max_val = tl.max(val, axis=0)
    val_minus_max = val - max_val
    exp_val = tl.exp(val_minus_max)
    sum_exp = tl.sum(exp_val, axis=0)
    
    # Compute softmax output
    softmax_out = exp_val / sum_exp
    
    # Store result
    output_ptrs = output_ptr + row_idx * n_elements + col_offsets
    tl.store(output_ptrs, softmax_out, mask=col_offsets < n_elements)


@torch.fx.wrap
def fused_attention_softmax(mask: torch.Tensor, attn: torch.Tensor, route: str = "") -> torch.Tensor:
    """
    Fused attention softmax kernel wrapper.
    
    mask: [1, 1, N, N] attention mask with -inf values
    attn: [1, H, N, N] attention scores
    """
    # Get dimensions
    batch_size, num_heads, seq_len, _ = attn.shape
    
    # Total output elements
    n_elements = seq_len * seq_len
    n_rows = batch_size * num_heads * seq_len
    
    # Allocate output
    output = torch.empty_like(attn)
    
    # Choose block size based on sequence length
    BLOCK_SIZE = 128 if seq_len <= 128 else 256
    
    # Launch kernel
    num_programs = n_rows
    
    fused_attention_softmax_kernel[(num_programs,)](
        mask_ptr=mask,
        attn_ptr=attn,
        output_ptr=output,
        n_elements=seq_len,
        n_rows=seq_len,
        row_stride=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1):
    """
    Match the attention computation pattern:
    1. Add mask to attention scores
    2. Max with -inf (no-op, can be eliminated)
    3. View reshape
    4. Softmax
    5. Dropout (disabled)
    
    Returns the final output.
    """
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=torch.device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(in_1.shape[0], in_1.shape[1], in_1.shape[2], in_1.shape[3])
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_attention_softmax