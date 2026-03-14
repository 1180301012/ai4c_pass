import torch
import triton
import triton.language as tl
import math

# Pattern matching function for BERT-style SDPA (positional arguments)
# Matches bert batch=1: view(1, -1, 2, 64), reshape(1, 64, 128)
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Matches: linear -> view -> transpose -> SDPA (positional) -> transpose -> reshape
    """
    tmp_2 = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = tmp_2.view(1, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 64, 128)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def fused_linear_transpose_kernel_pos(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, seq_len, in_features, num_heads, head_dim,
    input_stride_b, input_stride_s, input_stride_f,
    weight_stride_o, weight_stride_i,
    output_stride_b, output_stride_h, output_stride_s, output_stride_d,
    BLOCK_SEQ: tl.constexpr, BLOCK_DIM: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: linear + view + transpose
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_d = tl.arange(0, BLOCK_DIM)
    
    acc = tl.zeros([BLOCK_SEQ, BLOCK_DIM], dtype=tl.float32)
    
    for k_start in range(0, in_features, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < in_features
        
        input_offs = pid_b * input_stride_b + offs_s[:, None] * input_stride_s + offs_k[None, :] * input_stride_f
        input_mask = (offs_s[:, None] < seq_len) & k_mask[None, :]
        inp = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)
        
        weight_row = pid_h * head_dim + offs_d
        weight_offs = weight_row[None, :] * weight_stride_o + offs_k[:, None] * weight_stride_i
        weight_mask = k_mask[:, None] & (offs_d[None, :] < head_dim)
        w = tl.load(weight_ptr + weight_offs, mask=weight_mask, other=0.0)
        
        acc += tl.dot(inp, w)
    
    bias_offs = pid_h * head_dim + offs_d
    bias_mask = offs_d < head_dim
    b = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
    acc = acc + b[None, :]
    
    output_offs = (pid_b * output_stride_b + pid_h * output_stride_h + 
                   offs_s[:, None] * output_stride_s + offs_d[None, :] * output_stride_d)
    output_mask = (offs_s[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(output_ptr + output_offs, acc, mask=output_mask)


@triton.jit
def attention_fwd_kernel_pos(
    Q, K, V, mask_ptr, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch, num_heads, seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Standard attention: softmax(Q @ K.T * scale + mask) @ V
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    q_offs = pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    q = tl.load(Q + q_offs, mask=q_mask, other=0.0)
    
    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        k_offs = pid_b * stride_kb + pid_h * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        k_mask = (offs_n[None, :] < seq_len) & (offs_k[:, None] < head_dim)
        k = tl.load(K + k_offs, mask=k_mask, other=0.0)
        
        qk = tl.dot(q, k) * scale
        
        mask_offs = pid_b * stride_mb + pid_h * stride_mh + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
        mask_valid = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        attn_mask = tl.load(mask_ptr + mask_offs, mask=mask_valid, other=0.0)
        qk = qk + attn_mask
        
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        
        v_offs = pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v_mask = (offs_n[:, None] < seq_len) & (offs_k[None, :] < head_dim)
        v = tl.load(V + v_offs, mask=v_mask, other=0.0)
        
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        m_i = m_new
        l_i = l_new
    
    acc = acc / l_i[:, None]
    
    out_offs = pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_mask = (offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim)
    tl.store(Out + out_offs, acc, mask=out_mask)


@triton.jit  
def reshape_output_kernel_pos(
    input_ptr, output_ptr,
    batch, num_heads, seq_len, head_dim,
    in_stride_b, in_stride_h, in_stride_s, in_stride_d,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose [batch, heads, seq, dim] -> [batch, seq, heads*dim]
    """
    pid = tl.program_id(0)
    total = batch * seq_len * num_heads * head_dim
    
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    
    hidden_dim = num_heads * head_dim
    b = offs // (seq_len * hidden_dim)
    rem = offs % (seq_len * hidden_dim)
    s = rem // hidden_dim
    hd = rem % hidden_dim
    h = hd // head_dim
    d = hd % head_dim
    
    in_offs = b * in_stride_b + h * in_stride_h + s * in_stride_s + d * in_stride_d
    vals = tl.load(input_ptr + in_offs, mask=mask, other=0.0)
    
    out_offs = b * out_stride_b + s * out_stride_s + hd * out_stride_h
    tl.store(output_ptr + out_offs, vals, mask=mask)


@torch.fx.wrap
def fused_attention_positional(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Full fused attention implementation for BERT-style
    in_0: bias, in_1: weight, in_2: mask, in_3: hidden_states, in_4: key, in_5: query
    """
    batch = in_5.size(0)
    num_heads = in_5.size(1)
    seq_len = in_5.size(2)
    head_dim = in_5.size(3)
    in_features = in_3.size(-1)
    hidden_dim = num_heads * head_dim
    
    hidden_states = in_3.contiguous()
    weight = in_1.contiguous()
    bias = in_0.contiguous()
    query = in_5.contiguous()
    key = in_4.contiguous()
    mask = in_2.contiguous()
    
    # Step 1: Linear + view + transpose for value
    value = torch.empty(batch, num_heads, seq_len, head_dim, 
                        device=hidden_states.device, dtype=hidden_states.dtype)
    
    BLOCK_SEQ = 64
    BLOCK_DIM = 64
    BLOCK_K = 64
    
    grid_linear = (batch, num_heads, triton.cdiv(seq_len, BLOCK_SEQ))
    fused_linear_transpose_kernel_pos[grid_linear](
        hidden_states, weight, bias, value,
        batch, seq_len, in_features, num_heads, head_dim,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        weight.stride(0), weight.stride(1),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_DIM=BLOCK_DIM, BLOCK_K=BLOCK_K,
    )
    
    # Step 2: Attention computation
    attn_out = torch.empty_like(query)
    scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid_attn = (batch, num_heads, triton.cdiv(seq_len, BLOCK_M))
    attention_fwd_kernel_pos[grid_attn](
        query, key, value, mask, attn_out,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
        batch, num_heads, seq_len, head_dim,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=head_dim,
    )
    
    # Step 3: Reshape output [batch, heads, seq, dim] -> [batch, seq, hidden]
    output = torch.empty(batch, seq_len, hidden_dim, 
                         device=attn_out.device, dtype=attn_out.dtype)
    
    total_elements = batch * seq_len * hidden_dim
    BLOCK_SIZE = 1024
    grid_reshape = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    reshape_output_kernel_pos[grid_reshape](
        attn_out, output,
        batch, num_heads, seq_len, head_dim,
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_attention_positional