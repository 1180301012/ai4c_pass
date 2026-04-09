import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_0, in_4):
    """Pattern: Final tensor operations - unsqueeze expand and subtraction"""
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_9 = tmp_5.unsqueeze(3)
    tmp_10 = tmp_8 - tmp_6
    return (tmp_10, tmp_9)

def replacement_args(tmp_5, in_0, in_4):
    return (tmp_5, in_0, in_4)

@triton.jit
def final_ops_kernel(
    attention_ptr, in0_ptr, in4_ptr,
    out_sub_ptr, out_att_ptr,
    batch, seq, codewords, feats,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for final tensor operations"""
    # Calculate program indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    codeword_idx = tl.program_id(2)
    feat_idx = tl.program_id(3)
    
    # For expanded tensor from in4: [1,4096,32,512]
    expanded_offset = batch_idx * seq * codewords * feats + seq_idx * codewords * feats + codeword_idx * feats + feat_idx
    
    # For reshaped in0: [32,512] -> [1,1,32,512] 
    in0_offset = codeword_idx * feats + feat_idx
    
    # For attention with added dimension: [batch,seq,codewords] -> [batch,seq,codewords,1]
    attention_offset = batch_idx * seq * codewords + seq_idx * codewords + codeword_idx
    
    # Load values - processing one element at a time for simplicity
    expanded_val = tl.load(in4_ptr + expanded_offset)
    in0_val = tl.load(in0_ptr + in0_offset)
    attention_val = tl.load(attention_ptr + attention_offset)
    
    # Perform the subtraction
    sub_result = expanded_val - in0_val
    
    # Store results
    tl.store(out_sub_ptr + expanded_offset, sub_result)
    # Store attention value (broadcasting it to this feature position)
    tl.store(out_att_ptr + expanded_offset, attention_val)

@torch.fx.wrap
def optimized_final_operations(attention, in0, in4):
    """Wrapper for optimized final tensor operations"""
    # Get shapes
    batch, seq, codewords = attention.shape
    in0_shape = in0.shape  # [32, 512]
    in4_shape = in4.shape  # [1, 4096, 512]
    
    feats = in0_shape[1]  # Should be 512
    
    # Output shapes
    sub_shape = (1, in4_shape[1], in0_shape[0], in0_shape[1])  # [1,4096,32,512]
    att_shape = (1, in4_shape[1], in0_shape[0], in0_shape[1])  # [1,4096,32,512] with expanded dim
    
    output_sub = torch.empty(sub_shape, dtype=in4.dtype, device=in4.device)
    output_att = torch.empty(att_shape, dtype=attention.dtype, device=attention.device)
    
    # Launch kernel for each element in the expanded tensor
    # Grid: [batch, seq, codewords, feats]
    grid = (1, seq, codewords, feats)
    
    BLOCK_SIZE = 256
    
    final_ops_kernel[grid](
        attention_ptr=attention,
        in0_ptr=in0,
        in4_ptr=in4,
        out_sub_ptr=output_sub,
        out_att_ptr=output_att,
        batch=1, seq=seq, codewords=codewords, feats=feats,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (output_sub, output_att)

def replacement_func():
    return optimized_final_operations