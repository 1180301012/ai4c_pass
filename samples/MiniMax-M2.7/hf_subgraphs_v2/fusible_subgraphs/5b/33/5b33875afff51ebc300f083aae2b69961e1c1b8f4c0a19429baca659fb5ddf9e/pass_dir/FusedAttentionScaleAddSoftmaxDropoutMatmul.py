import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel_8(
    scores_ptr, mask_ptr, value_ptr, output_ptr,
    scale: tl.constexpr,
    batch: tl.constexpr, heads: tl.constexpr, 
    seq_len: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused attention kernel with scale=8.0"""
    pid = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    batch_idx = pid // seq_len
    seq_out_idx = pid % seq_len
    
    offs = tl.arange(0, BLOCK_SIZE)
    
    max_val = float("-inf")
    for start in range(0, seq_len, BLOCK_SIZE):
        mask_s = (start + offs) < seq_len
        s_offset = (batch_idx * heads * seq_len * seq_len + 
                    head_idx * seq_len * seq_len + 
                    seq_out_idx * seq_len + start)
        s = tl.load(scores_ptr + s_offset + offs, mask=mask_s, other=0.0)
        m = tl.load(mask_ptr + offs, mask=mask_s, other=0.0)
        scaled_masked = (s / scale) + m
        max_val = tl.max(max_val, tl.max(scaled_masked))
    
    exp_sum = 0.0
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for hd_start in range(0, head_dim, BLOCK_SIZE):
        hd_offs = hd_start + offs
        mask_h = hd_offs < head_dim
        acc_hd = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for start in range(0, seq_len, BLOCK_SIZE):
            mask_s = (start + offs) < seq_len
            s_offset = (batch_idx * heads * seq_len * seq_len + 
                        head_idx * seq_len * seq_len + 
                        seq_out_idx * seq_len + start)
            s = tl.load(scores_ptr + s_offset + offs, mask=mask_s, other=0.0)
            m = tl.load(mask_ptr + offs, mask=mask_s, other=0.0)
            exp_val = tl.exp((s / scale) + m - max_val)
            exp_sum = exp_sum + tl.sum(exp_val)
        
        for k in range(seq_len):
            v_offset = (batch_idx * heads * seq_len * head_dim + 
                        head_idx * seq_len * head_dim + 
                        k * head_dim + hd_start)
            v = tl.load(value_ptr + v_offset + offs * head_dim, mask=mask_h, other=0.0)
            
            for start in range(0, seq_len, BLOCK_SIZE):
                mask_s = (start + offs) < seq_len
                s_offset = (batch_idx * heads * seq_len * seq_len + 
                            head_idx * seq_len * seq_len + 
                            seq_out_idx * seq_len + start)
                s = tl.load(scores_ptr + s_offset + offs, mask=mask_s, other=0.0)
                m = tl.load(mask_ptr + offs, mask=mask_s, other=0.0)
                exp_val = tl.exp((s / scale) + m - max_val)
                acc_hd = acc_hd + exp_val[k] * v
        
        acc = acc + acc_hd
    
    output = acc / exp_sum
    
    out_offset = (batch_idx * seq_len * heads * head_dim +
                  seq_out_idx * heads * head_dim +
                  head_idx * head_dim)
    tl.store(output_ptr + out_offset + offs, output, mask=offs < head_dim)


@triton.jit
def fused_attention_kernel_2(
    scores_ptr, mask_ptr, value_ptr, output_ptr,
    scale: tl.constexpr,
    batch: tl.constexpr, heads: tl.constexpr, 
    seq_len: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused attention kernel with scale=2.828..."""
    pid = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    batch_idx = pid // seq_len
    seq_out_idx = pid % seq_len
    
    offs = tl.arange(0, BLOCK_SIZE)
    
    max_val = float("-inf")
    for start in range(0, seq_len, BLOCK_SIZE):
        mask_s = (start + offs) < seq_len
        s_offset = (batch_idx * heads * seq_len * seq_len + 
                    head_idx * seq_len * seq_len + 
                    seq_out_idx * seq_len + start)
        s = tl.load(scores_ptr + s_offset + offs, mask=mask_s, other=0.0)
        m = tl.load(mask_ptr + offs, mask=mask_s, other=0.0)
        scaled_masked = (s / scale) + m
        max_val = tl.max(max_val, tl.max(scaled_masked))
    
    exp_sum = 0.0
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for hd_start in range(0, head_dim, BLOCK_SIZE):
        hd_offs = hd_start + offs
        mask_h = hd_offs < head_dim
        acc_hd = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for start in range(0, seq_len, BLOCK_SIZE):
            mask_s = (start + offs) < seq_len
            s_offset = (batch_idx * heads * seq_len * seq_len + 
                        head_idx * seq_len * seq_len + 
                        seq_out_idx * seq_len + start)
            s = tl.load(scores_ptr + s_offset + offs, mask=mask_s, other=0.0)
            m = tl.load(mask_ptr + offs, mask=mask_s, other=0.0)
            exp_val = tl.exp((s / scale) + m - max_val)
            exp_sum = exp_sum + tl.sum(exp_val)
        
        for k in range(seq_len):
            v_offset = (batch_idx * heads * seq_len * head_dim + 
                        head_idx * seq_len * head_dim + 
                        k * head_dim + hd_start)
            v = tl.load(value_ptr + v_offset + offs * head_dim, mask=mask_h, other=0.0)
            
            for start in range(0, seq_len, BLOCK_SIZE):
                mask_s = (start + offs) < seq_len
                s_offset = (batch_idx * heads * seq_len * seq_len + 
                            head_idx * seq_len * seq_len + 
                            seq_out_idx * seq_len + start)
                s = tl.load(scores_ptr + s_offset + offs, mask=mask_s, other=0.0)
                m = tl.load(mask_ptr + offs, mask=mask_s, other=0.0)
                exp_val = tl.exp((s / scale) + m - max_val)
                acc_hd = acc_hd + exp_val[k] * v
        
        acc = acc + acc_hd
    
    output = acc / exp_sum
    
    out_offset = (batch_idx * seq_len * heads * head_dim +
                  seq_out_idx * heads * head_dim +
                  head_idx * head_dim)
    tl.store(output_ptr + out_offset + offs, output, mask=offs < head_dim)


def pattern_scale8(in_0, in_2, in_3):
    """Match attention pattern with scale 8.0"""
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args_scale8(in_0, in_2, in_3):
    return (in_0, in_2, in_3, "scale8")


@torch.fx.wrap
def triton_fused_attention_dispatch(in_0, in_2, in_3, route):
    """Dispatch to appropriate kernel based on route string."""
    batch, heads, seq_len, _ = in_0.shape
    _, _, _, head_dim = in_3.shape
    
    output = torch.empty(batch, seq_len, heads, head_dim, 
                        device=in_0.device, dtype=in_0.dtype)
    
    BLOCK_SIZE = 128
    grid = (batch * seq_len, heads)
    
    if route == "scale8":
        fused_attention_kernel_8[grid](
            in_0, in_2, in_3, output,
            scale=8.0,
            batch=batch, heads=heads,
            seq_len=seq_len, head_dim=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif route == "scale2":
        fused_attention_kernel_2[grid](
            in_0, in_2, in_3, output,
            scale=2.8284271247461903,
            batch=batch, heads=heads,
            seq_len=seq_len, head_dim=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output


def replacement_func():
    return triton_fused_attention_dispatch