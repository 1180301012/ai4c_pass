import torch
import triton
import triton.language as tl


@torch.fx.wrap
def fused_patch_embed_layer_norm(
    cls_token,          # [1, 1, 768]
    patch_emb,          # [N, seq_len, 768] after transpose  
    pos_emb,            # [1, 981, 768]
    ln_weight,          # [768]
    ln_bias,            # [768]
):
    """
    Fused kernel for:
    tmp_9 = cls_token.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + pos_emb
    tmp_12 = dropout(tmp_11, 0.0, ...)
    tmp_13 = layer_norm(tmp_12, ...)
    
    Returns tmp_12 and tmp_13
    """
    N = patch_emb.shape[0]
    seq_len = patch_emb.shape[1]
    hidden_dim = patch_emb.shape[2]
    total_seq = 981  # 1 (cls) + 980 (patches)
    
    # Allocate outputs
    add_out = torch.empty((N, total_seq, hidden_dim), dtype=patch_emb.dtype, device=patch_emb.device)
    ln_out = torch.empty_like(add_out)
    
    BLOCK_SIZE = 256
    num_elements = N * total_seq * hidden_dim
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Kernel 1: Compute add_out = cls + patch_emb + pos_emb
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    @triton.jit
    def add_kernel(
        cls_ptr, patch_ptr, pos_ptr, out_ptr,
        N: tl.constexpr, seq_len: tl.constexpr, hidden: tl.constexpr,
        total_seq: tl.constexpr, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < N * total_seq * hidden
        
        row = offsets // hidden
        col = offsets % hidden
        
        # cls_token is [1, 1, 768], broadcasted - load from offset col
        cls_val = tl.load(cls_ptr + col, mask=mask, other=0.0)
        
        # patch_emb is [N, seq_len, hidden]
        # map global row to (batch, patch_idx)
        batch = row // seq_len
        patch_idx = row % seq_len
        patch_offset = (batch * seq_len + patch_idx) * hidden + col
        patch_val = tl.load(patch_ptr + patch_offset, mask=mask, other=0.0)
        
        # pos_emb is [1, total_seq, hidden], row broadcasts
        pos_val = tl.load(pos_ptr + row * hidden + col, mask=mask, other=0.0)
        
        result = cls_val + patch_val + pos_val
        tl.store(out_ptr + offsets, result, mask=mask)
    
    add_kernel[grid](
        cls_token, patch_emb, pos_emb, add_out,
        N, seq_len, hidden_dim, total_seq, BLOCK_SIZE
    )
    
    # Kernel 2: Compute layer_norm per row
    @triton.jit
    def layer_norm_kernel(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        N: tl.constexpr, total_seq: tl.constexpr, hidden: tl.constexpr,
        BLOCK: tl.constexpr
    ):
        row_pid = tl.program_id(0)
        
        # Compute mean and var for this row
        base_offset = row_pid * hidden
        
        sum_val = 0.0
        sum_sq = 0.0
        
        # Sequential reduce
        for i in range(hidden):
            offset = base_offset + i
            val = tl.load(x_ptr + offset, mask=row_pid * hidden + i < N * total_seq * hidden, other=0.0)
            sum_val += val
            sum_sq += val * val
        
        mean = sum_val / hidden
        var = (sum_sq / hidden) - (mean * mean)
        inv_std = tl.rsqrt(var + 1e-06)
        
        # Compute output
        for i in range(hidden):
            offset = base_offset + i
            mask_val = row_pid * hidden + i < N * total_seq * hidden
            val = tl.load(x_ptr + offset, mask=mask_val, other=0.0)
            ln_val = (val - mean) * inv_std
            w = tl.load(weight_ptr + i, mask=i < hidden, other=1.0)
            b = tl.load(bias_ptr + i, mask=i < hidden, other=0.0)
            out = ln_val * w + b
            tl.store(out_ptr + offset, out, mask=mask_val)
    
    num_rows = N * total_seq
    layer_norm_kernel[(num_rows,)](
        add_out, ln_weight, ln_bias, ln_out,
        N, total_seq, hidden_dim, BLOCK_SIZE
    )
    
    return add_out, ln_out


def pattern(in_2, tmp_8, in_3, in_5, in_4):
    """
    Match the pattern from the original model:
    tmp_9 = in_2.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim = 1)
    tmp_11 = tmp_10 + in_3
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_5, in_4, 1e-06)
    
    Returns tmp_12 and tmp_13
    """
    tmp_9 = in_2.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    computed_add = tmp_10 + in_3
    dropout_out = torch.nn.functional.dropout(computed_add, 0.0, False, False)
    ln_out = torch.nn.functional.layer_norm(dropout_out, (768,), in_5, in_4, 1e-06)
    return dropout_out, ln_out


def replacement_args(in_2, tmp_8, in_3, in_5, in_4):
    """
    Extract arguments needed for the fused kernel.
    """
    return (in_2, tmp_8, in_3, in_5, in_4)


def replacement_func():
    return fused_patch_embed_layer_norm