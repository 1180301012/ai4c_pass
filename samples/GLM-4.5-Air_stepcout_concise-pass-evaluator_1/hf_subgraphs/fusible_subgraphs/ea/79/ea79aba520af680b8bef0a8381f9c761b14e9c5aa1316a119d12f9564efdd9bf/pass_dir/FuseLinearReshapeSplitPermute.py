import torch
import triton
import triton.language as tl

def pattern(in_3, in_2, in_1):
    """
    Pattern matching: Linear transformation + Reshape + Split + Multiple Permute operations
    This pattern matches the core computation from the model:
    1. Linear: linear(in_3, in_2, in_1) -> [batch_size, 49, 1536]
    2. Reshape: reshape(batch_size, 49, 8, -1) -> [batch_size, 49, 8, 192]
    3. Split: split([32, 32, 128], dim=3) -> 3 tensors [batch_size, 49, 8, *]
    4. Permute: permute(0, 2, 1, 3) on each split -> [batch_size, 8, 49, *]
    5. Transpose: transpose(-2, -1) on middle tensor -> [batch_size, 8, 32, 49]
    """
    tmp_3 = torch.nn.functional.linear(in_3, in_2, in_1)
    tmp_4 = tmp_3.reshape(in_3.shape[0], 49, 8, -1)
    tmp_5 = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_13 = tmp_10.transpose(-2, -1)
    return (tmp_9, tmp_13, tmp_11)

def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)

@triton.jit
def fused_kernel_3d_batched(
    x_ptr,                   # Input tensor [batch_size, seq_len, in_features]
    weight_ptr,             # Weight [out_features, in_features]
    bias_ptr,               # Bias [out_features]
    out_q_ptr,              # Output Q [batch_size, 8, seq_len, 32]
    out_k_ptr,              # Output K [batch_size, 8, 32, seq_len]
    out_v_ptr,              # Output V [batch_size, 8, seq_len, 128]
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEATURES: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    """Fused kernel that performs linear transformation + reshape + split + permute + transpose"""
    
    # Compute program indices
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_feature_q = tl.program_id(2)  # Q/K/V features
    pid_feature_qk = tl.program_id(3)  # For Q/K which have smaller features
    
    # Batch and sequence constraints
    if pid_batch >= batch_size or pid_seq >= seq_len:
        return
    
    # Determine which output we're processing (Q, K, or V)
    if pid_feature_q < 2:  # Q or K (32 features each)
        features_out = 32
        base_feature = pid_feature_q * 32
        is_q = (pid_feature_q == 0)
        is_k = (pid_feature_q == 1)
    else:  # V (128 features)
        features_out = 128
        base_feature = 64 + (pid_feature_q - 2) * 128
        is_q = False
        is_k = False
        is_v = True
    
    # Compute offsets and masks
    if is_k:
        # For K: [batch, 8, 32, seq_len]
        batch_offset = pid_batch * seq_len * features_out
        seq_offset = pid_seq * features_out
        offset_k = batch_offset + 8 * seq_offset + pid_feature_qk * seq_len * features_out
        mask_k = pid_feature_qk < 32
        
        # Load bias for this output channel
        bias_val = tl.load(bias_ptr + base_feature, eviction_policy='evict_last')
        
        # Compute linear transformation for this K element
        acc = bias_val
        for f_idx in range(0, in_features, BLOCK_SIZE_FEATURES):
            f_offset = f_idx + tl.arange(0, BLOCK_SIZE_FEATURES)
            f_mask = f_offset < in_features
            
            # Load input and weight
            x_val = tl.load(x_ptr + pid_batch * seq_len * in_features + pid_seq * in_features + f_offset,
                           mask=f_mask, other=0.0)
            w_val = tl.load(weight_ptr + base_feature * in_features + f_offset,
                           mask=f_mask, other=0.0)
            acc += x_val * w_val
        
        # Store K result
        tl.store(out_k_ptr + offset_k, acc, mask=mask_k)
    
    elif is_q:
        # For Q: [batch, 8, seq_len, 32]
        head_idx = pid_feature_qk // seq_len
        seq_idx = pid_feature_qk % seq_len
        
        batch_offset = pid_batch * seq_len * 32
        seq_offset = seq_idx * 32
        offset_q = batch_offset + 8 * seq_offset + head_idx * 32
        mask_q = True  # No bounds check needed for computed indices
        
        # Load bias
        bias_val = tl.load(bias_ptr + base_feature, eviction_policy='evict_last')
        
        # Compute linear transformation
        acc = bias_val
        for f_idx in range(0, in_features, BLOCK_SIZE_FEATURES):
            f_offset = f_idx + tl.arange(0, BLOCK_SIZE_FEATURES)
            f_mask = f_offset < in_features
            
            x_val = tl.load(x_ptr + pid_batch * seq_len * in_features + seq_idx * in_features + f_offset,
                           mask=f_mask, other=0.0)
            w_val = tl.load(weight_ptr + base_feature * in_features + f_offset,
                           mask=f_mask, other=0.0)
            acc += x_val * w_val
        
        # Store Q result
        tl.store(out_q_ptr + offset_q, acc, mask=mask_q)
    
    elif is_v:
        # For V: [batch, 8, seq_len, 128]
        head_idx = pid_feature_qk // seq_len
        seq_idx = pid_feature_qk % seq_len
        
        batch_offset = pid_batch * seq_len * 128
        seq_offset = seq_idx * 128
        offset_v = batch_offset + 8 * seq_offset + head_idx * 128
        mask_v = True
        
        # Load bias
        bias_val = tl.load(bias_ptr + base_feature, eviction_policy='evict_last')
        
        # Compute linear transformation
        acc = bias_val
        for f_idx in range(0, in_features, BLOCK_SIZE_FEATURES):
            f_offset = f_idx + tl.arange(0, BLOCK_SIZE_FEATURES)
            f_mask = f_offset < in_features
            
            x_val = tl.load(x_ptr + pid_batch * seq_len * in_features + seq_idx * in_features + f_offset,
                           mask=f_mask, other=0.0)
            w_val = tl.load(weight_ptr + base_feature * in_features + f_offset,
                           mask=f_mask, other=0.0)
            acc += x_val * w_val
        
        # Store V result
        tl.store(out_v_ptr + offset_v, acc, mask=mask_v)

@torch.fx.wrap
def fused_linear_reshape_split_permute_cuda(in_3, in_2, in_1):
    """
    Wrapper function that launches the fused kernel
    """
    batch_size, seq_len, in_features = in_3.shape
    out_features = in_2.shape[0]
    
    # Q: [batch, 8, seq, 32], K: [batch, 8, 32, seq], V: [batch, 8, seq, 128]
    out_q = torch.empty((batch_size, 8, seq_len, 32), dtype=in_3.dtype, device=in_3.device)
    out_k = torch.empty((batch_size, 8, 32, seq_len), dtype=in_3.dtype, device=in_3.device)
    out_v = torch.empty((batch_size, 8, seq_len, 128), dtype=in_3.dtype, device=in_3.device)
    
    # Grid configuration: (batch_size, seq_len, features_per_output_type, additional_dimension)
    if out_features == 1536:
        # Split into Q[32], K[32], V[128]
        grid = lambda meta: (
            batch_size,
            seq_len,
            3,  # 3 output types
            max(32, seq_len)  # Max dimension for additional split
        )
        
        fused_kernel_3d_batched[grid](
            in_3, in_2, in_1,
            out_q, out_k, out_v,
            batch_size, seq_len, in_features, out_features,
            BLOCK_SIZE_BATCH=1,
            BLOCK_SIZE_FEATURES=256,
            BLOCK_SIZE_SEQ=1
        )
        
        return out_q, out_k, out_v
    else:
        # Fallback to original implementation for unsupported output sizes
        tmp_3 = torch.nn.functional.linear(in_3, in_2, in_1)
        tmp_4 = tmp_3.reshape(batch_size, seq_len, 8, -1)
        tmp_5 = tmp_4.split([32, 32, 128], dim=3)
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_8 = tmp_5[2]
        tmp_9 = tmp_6.permute(0, 2, 1, 3)
        tmp_10 = tmp_7.permute(0, 2, 1, 3)
        tmp_11 = tmp_8.permute(0, 2, 1, 3)
        tmp_13 = tmp_10.transpose(-2, -1)
        return tmp_9, tmp_13, tmp_11

def replacement_func():
    return fused_linear_reshape_split_permute_cuda