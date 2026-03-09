import torch
import triton
import triton.language as tl

def pattern(cls_token, flatten_transposed_out):
    # Matches: expand -> cat -> dropout (p=0, which is a no-op)
    tmp_8 = cls_token.expand(1, -1, -1)
    tmp_9 = torch.cat((tmp_8, flatten_transposed_out), dim=1)
    tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
    return tmp_10

def replacement_args(cls_token, flatten_transposed_out):
    return (cls_token, flatten_transposed_out)

@triton.jit
def fused_expand_cat_kernel(
    cls_token_ptr,
    flatten_transposed_ptr,
    output_ptr,
    cls_batch,
    cls_seq,
    cls_features,
    flatten_batch,
    flatten_seq,
    flatten_features,
    total_features,
    total_seq,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = flatten_batch * total_features * total_seq
    
    # Calculate offsets for parallel processing
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    batch = offsets // (total_features * total_seq)
    rem = offsets % (total_features * total_seq)
    feature = rem // total_seq
    seq = rem % total_seq
    
    if feature < cls_features:
        # Load from cls_token expanded
        cls_value = tl.load(cls_token_ptr + 0, mask=mask, other=0.0)  # All cls tokens are same
        output_offset = batch * total_features * total_seq + feature * total_seq + seq
        tl.store(output_ptr + output_offset, cls_value, mask=mask)
    else:
        # Load from flatten_transposed_out
        flatten_feature_idx = feature - cls_features
        flatten_offset = batch * flatten_features * flatten_seq + flatten_feature_idx * flatten_seq + seq
        x = tl.load(flatten_transposed_ptr + flatten_offset, mask=mask, other=0.0)
        output_offset = batch * total_features * total_seq + feature * total_seq + seq
        tl.store(output_ptr + output_offset, x, mask=mask)

@torch.fx.wrap
def fused_expand_cat_dropout_gpu(cls_token, flatten_transposed_out):
    # Input shapes:
    # cls_token: [1, 1, 768] -> will be expanded to [1, seq_len, 768]
    # flatten_transposed_out: [1, 768, seq_len]
    # Output should be: [1, 768+seq_len, seq_len] with dropout (no-op)
    
    cls_batch, cls_seq, cls_features = cls_token.shape
    flatten_batch, flatten_features, flatten_seq = flatten_transposed_out.shape
    
    # Since cls_token is [1, 1, 768] and we expand to [1, -1, 768], 
    # the expanded sequence length should match flatten_seq
    if cls_seq == 1:
        expanded_seq = flatten_seq
    else:
        expanded_seq = cls_seq
    
    # Final concatenation on dim=1: [1, 768 + 768, seq_len]
    total_features = cls_features + flatten_features  # 768 + 768 = 1536
    total_seq = flatten_seq
    
    output = torch.empty(cls_batch, total_features, total_seq, 
                        dtype=cls_token.dtype, device=cls_token.device)
    
    N = cls_batch * total_features * total_seq
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_expand_cat_kernel[(num_programs,)](
        cls_token_ptr=cls_token,
        flatten_transposed_ptr=flatten_transposed_out,
        output_ptr=output,
        cls_batch=cls_batch,
        cls_seq=cls_seq,
        cls_features=cls_features,
        flatten_batch=flatten_batch,
        flatten_seq=flatten_seq,
        flatten_features=flatten_features,
        total_features=total_features,
        total_seq=total_seq,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_expand_cat_dropout_gpu