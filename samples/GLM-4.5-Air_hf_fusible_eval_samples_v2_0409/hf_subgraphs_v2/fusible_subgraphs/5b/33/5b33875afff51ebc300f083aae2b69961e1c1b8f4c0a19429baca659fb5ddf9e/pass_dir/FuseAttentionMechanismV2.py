import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel_v2(
    attention_scores_ptr, value_layer_ptr, attention_mask_ptr,
    output_ptr, scale_factor,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program ID for this block
    pid = tl.program_id(0)
    
    # Compute offset for this program
    offset = pid * BLOCK_SIZE_M * seq_len * head_dim
    m_offset = (pid // ((seq_len * head_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)) * BLOCK_SIZE_M
    n_offset = (pid % ((seq_len * head_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)) * BLOCK_SIZE_M
    
    # Create mask for valid indices
    mask = tl.arange(0, BLOCK_SIZE_M) < (seq_len - m_offset)
    n_mask = tl.arange(0, BLOCK_SIZE_M) < (head_dim - n_offset)
    
    # Load attention scores
    scores = tl.load(
        attention_scores_ptr + offset + tl.arange(0, BLOCK_SIZE_M * seq_len),
        mask=mask[:, None],
        other=0.0
    ).reshape(BLOCK_SIZE_M, seq_len)
    
    # Load attention mask (broadcast)
    mask_tensor = tl.load(
        attention_mask_ptr + m_offset * seq_len,
        mask=mask,
        other=0.0
    )
    mask_tensor = tl.broadcast_to(mask_tensor[:, None], (BLOCK_SIZE_M, seq_len))
    
    # Load values
    values = tl.load(
        value_layer_ptr + offset + tl.arange(0, BLOCK_SIZE_M * head_dim),
        mask=mask[:, None] & n_mask[None, :],
        other=0.0
    ).reshape(BLOCK_SIZE_M, head_dim)
    
    # Apply scale factor and attention mask
    scaled_scores = scores * scale_factor
    masked_scores = scaled_scores + mask_tensor
    
    # Simplified softmax computation
    max_scores = tl.max(masked_scores, axis=1, keepdim=True)
    exp_scores = tl.exp(masked_scores - max_scores)
    sum_exp = tl.sum(exp_scores, axis=1, keepdim=True)
    attention_weights = exp_scores / (sum_exp + 1e-6)
    
    # Simplified dropout
    keep_prob = 0.9
    dropout_mask = tl.rand((BLOCK_SIZE_M, seq_len)) < keep_prob
    attention_weights = attention_weights * dropout_mask
    
    # Compute matmul
    output = tl.dot(attention_weights, values, out_dtype=tl.float32)
    
    # Apply transpose and contiguous
    output = output.transpose(0, 1).contiguous()
    
    # Store result
    tl.store(
        output_ptr + offset,
        output,
        mask=mask[:, None] & n_mask[None, :]
    )

@torch.fx.wrap
def fused_attention_forward_v2(attention_scores, value_layer, attention_mask, scale_factor):
    batch_size, num_heads, seq_len, _ = attention_scores.shape
    head_dim = value_layer.shape[-1]
    
    # Determine optimal block sizes
    BLOCK_SIZE_M = 128
    
    # Calculate grid size
    total_elements = seq_len * head_dim
    num_programs = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Create output tensor
    output = torch.empty((batch_size, num_heads, seq_len, head_dim), 
                        dtype=attention_scores.dtype, device=attention_scores.device)
    
    # Launch kernel for each head and batch
    for b in range(batch_size):
        for n in range(num_heads):
            fused_attention_kernel_v2[num_programs](
                attention_scores.data_ptr() + b * attention_scores.stride(0) + n * attention_scores.stride(1),
                value_layer.data_ptr() + b * value_layer.stride(0) + n * value_layer.stride(1),
                attention_mask.data_ptr() + b * attention_mask.stride(0),
                output.data_ptr() + b * output.stride(0) + n * output.stride(1),
                scale_factor,
                1, 1, seq_len, head_dim,
                BLOCK_SIZE_M, BLOCK_SIZE_M
            )
    
    return output

def pattern(in_0, in_2, in_3):
    # Pattern matches the exact computation from the model with the 2.828 scale factor
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(in_0, in_2, in_3):
    # Extract and pass all the arguments needed for the replacement
    scale_factor = 2.8284271247461903  # Scale factor used in this pattern
    return (in_0, in_3, in_2, scale_factor)

def replacement_func():
    return fused_attention_forward_v2