import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the full attention computation pattern:
    bmm(Q, K^T) -> softmax -> dropout (p=0, no-op) -> bmm -> view -> transpose -> reshape
    """
    # Step 1: Q @ K^T
    bmm = torch.bmm(in_0, in_1)
    
    # Step 2: softmax along last dimension
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    
    # Step 3: dropout (p=0.0, training=False is a no-op)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    
    # Step 4: attention weights @ V
    bmm_1 = torch.bmm(tmp_2, in_2)
    
    # Step 5: view to 4D [1, batch, 1, head_dim]
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    head_dim = in_0.shape[2]
    tmp_4 = bmm_1.view(1, batch_size, 1, head_dim)
    
    # Step 6: transpose [1, batch, 1, head_dim] -> [1, 1, batch, head_dim]
    tmp_5 = tmp_4.transpose(1, 2)
    
    # Step 7: reshape to [1, 1, batch * head_dim]
    tmp_6 = tmp_5.reshape(1, 1, batch_size * head_dim)
    
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    B: tl.constexpr, 
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused attention kernel: softmax(Q @ K^T) @ V -> reshape
    
    Input shapes:
    - Q (in_0): [B, 1, D]
    - K (in_1): [B, D, 1]
    - V (in_2): [B, 1, D]
    
    Output shape:
    - [1, 1, B * D]
    
    The computation is:
    1. score = Q @ K^T = [B, 1, 1] (scalar per batch)
    2. attn_weight = softmax(score, dim=-1) = score (since only 1 element)
    3. output = attn_weight * V = [B, 1, D]
    4. reshape output -> [1, 1, B*D]
    """
    out_elements = B * head_dim
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_elements
    
    # Compute (b, h) indices from flat output index
    b = offsets // head_dim
    h = offsets % head_dim
    
    # Step 1: Compute score = Q @ K^T = sum_d Q[b,0,d] * K[b,d,0] (scalar)
    # Q is [B, 1, D] stored as [B*D], K is [B, D, 1] stored as [B*D] (interleaved)
    # Actually K is [B, D, 1] in row-major: K[b,d,0] -> K[b*D + d, 0] if flattened
    
    # Load Q[b, 0, h] from in_0: [B, 1, D] -> flat index b * head_dim + h
    q_offset = b * head_dim + h
    q_val = tl.load(q_ptr + q_offset)
    
    # Load K[b, h, 0] from in_1: [B, D, 1] -> flat index b * head_dim + h  
    k_val = tl.load(k_ptr + q_offset)  # Same flat index structure
    
    # Compute score for this batch element
    score = q_val * k_val
    
    # Load V[b, 0, h] from in_2: [B, 1, D] -> flat index b * head_dim + h
    v_val = tl.load(v_ptr + q_offset)
    
    # Output = score * V
    out_val = score * v_val
    
    # Store to output [1, 1, B*D]
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_attention_wrapper(in_0, in_1, in_2):
    """
    Wrapper for fused attention kernel.
    
    Key insight: softmax over a single element always equals 1.0,
    so the attention weight is always 1.0.
    Output is simply V reshaped to [1, 1, B*D].
    
    Input shapes:
    - in_0 (Q): [B, 1, D]
    - in_1 (K^T): [B, D, 1]
    - in_2 (V): [B, 1, D]
    
    Output shape: [1, 1, B * D]
    """
    B = in_0.shape[0]  # batch size (e.g., 8 or 16)
    head_dim = in_0.shape[2]  # head dimension (e.g., 32 or 64)
    
    # Output size after final reshape
    out_elements = B * head_dim
    
    # Allocate output [1, 1, B*head_dim]
    out = torch.empty((1, 1, out_elements), dtype=in_0.dtype, device=in_0.device)
    
    # Use 1D grid where each program handles BLOCK_SIZE elements
    BLOCK_SIZE = 128
    num_programs = (out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_kernel[(num_programs,)](
        q_ptr=in_0,
        k_ptr=in_1,
        v_ptr=in_2,
        out_ptr=out,
        B=B,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_attention_wrapper