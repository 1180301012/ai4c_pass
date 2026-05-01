import torch
import triton
import triton.language as tl

@triton.jit
def sdpa_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len
    
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk, 
                mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    k = tl.load(K_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kk, 
                mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    dot = tl.dot(q, k, allow_tf32=True)
    softmax = tl.exp(dot - tl.max(dot, axis=1))
    softmax = softmax / tl.sum(softmax, axis=1)
    
    v = tl.load(V_ptr + offs_n[:, None] * stride_vm + offs_m[None, :] * stride_vk, 
                mask=mask_n[:, None] & mask_m[None, :], other=0.0)
    out = tl.dot(softmax, v, allow_tf32=True)
    
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok, 
             out, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def triton_sdpa(Q, K, V, attn_mask):
    
    batch, heads, seq_len, head_dim = Q.shape
    
    Q = Q.reshape(-1, seq_len, head_dim)
    K = K.reshape(-1, seq_len, head_dim)
    V = V.reshape(-1, seq_len, head_dim)
    
    out = torch.empty_like(Q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim
    
    grid = (tl.cdiv(seq_len, BLOCK_M), tl.cdiv(seq_len, BLOCK_N))
    
    sdpa_kernel[grid](
        Q, K, V, out,
        seq_len, head_dim,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
    
    return out.reshape(batch, heads, seq_len, head_dim)

def pattern(in_5, in_4, tmp_4, in_2):
    return torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)

def replacement_args(in_5, in_4, tmp_4, in_2):
    return (in_5, in_4, tmp_4, in_2)

def replacement_func():
    return triton_sdpa