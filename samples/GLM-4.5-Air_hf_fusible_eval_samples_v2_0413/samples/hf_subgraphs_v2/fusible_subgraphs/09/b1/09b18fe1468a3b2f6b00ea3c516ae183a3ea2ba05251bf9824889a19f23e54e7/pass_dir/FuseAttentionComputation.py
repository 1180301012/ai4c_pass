import torch
import triton
import triton.language as tl

@triton.jit
def attention_kernel(
    in_1_ptr,        # [1, 4096, 32, 512]
    in_2_ptr,        # [1, 1, 32, 512]
    in_3_ptr,        # [1, 1, 32] 
    out_ptr,         # [1, 4096, 32]
    stride_in1_batch, stride_in1_feat, stride_in1_inner,
    stride_in2_batch, stride_in2_feat, stride_in2_inner,
    stride_in3_batch, stride_in3_feat,
    stride_out_batch, stride_out_feat,
    N: tl.constexpr,   # 4096 (batch size)
    K: tl.constexpr,   # 32 (features) 
    D: tl.constexpr,   # 512 (feature dimension)
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= N:
        return
        
    # Calculate offsets for this sample
    sample_offset = pid * stride_out_batch
    feat_offsets = tl.arange(0, K) * stride_out_feat
    
    # Process features in chunks
    for feat_start in range(0, K, BLOCK_SIZE):
        feat_end = min(feat_start + BLOCK_SIZE, K)
        feat_count = feat_end - feat_start
        
        # Get current feature offsets
        current_feat_offsets = feat_offsets[feat_start:feat_start+feat_count]
        
        # Load scale (broadcasted)
        scale_offsets = sample_offset + current_feat_offsets * stride_in3_feat
        scale = tl.load(in_3_ptr + scale_offsets)
        
        # Initialize accumulator for sum of squared differences
        sum_sq = tl.zeros([feat_count], dtype=tl.float32)
        
        # Process feature dimension in chunks
        for d_start in range(0, D, 128):
            d_end = min(d_start + 128, D)
            d_count = d_end - d_start
            
            # Load in_1 data: [1, 4096, 32, 512]
            in1_offsets = sample_offset + current_feat_offsets[:, None] * stride_in1_feat + \
                         (d_start + tl.arange(0, d_count))[:, None] * stride_in1_inner
            in1_data = tl.load(in_1_ptr + in1_offsets, 
                              mask=(d_start + tl.arange(0, d_count))[:, None] < D, 
                              other=0.0)
            
            # Load in_2 data: [1, 1, 32, 512] (broadcasting)
            in2_offsets = current_feat_offsets[:, None] * stride_in2_feat + \
                         (d_start + tl.arange(0, d_count))[:, None] * stride_in2_inner
            in2_data = tl.load(in_2_ptr + in2_offsets, 
                              mask=(d_start + tl.arange(0, d_count))[:, None] < D, 
                              other=0.0)
            
            # Compute squared differences and accumulate
            diff = in1_data - in2_data
            diff_sq = diff * diff
            sum_sq += tl.sum(diff_sq, axis=1)
        
        # Apply scaling
        scaled = sum_sq * scale
        
        # Apply softmax
        max_val = tl.max(scaled)
        shifted = scaled - max_val
        exp_val = tl.exp(shifted)
        sum_exp = tl.sum(exp_val)
        softmax_result = exp_val / sum_exp
        
        # Store result
        out_offsets = sample_offset + current_feat_offsets * stride_out_feat
        tl.store(out_ptr + out_offsets, softmax_result)

@torch.fx.wrap
def fused_attention_computation(in_1, in_2, in_3):
    # Get tensor shapes and strides  
    N, K, D = in_1.shape[1], in_1.shape[2], in_1.shape[3]
    
    # Output shape [1, N, K]
    out = torch.empty((1, N, K), dtype=in_1.dtype, device=in_1.device)
    
    # Configure kernel - one program per batch sample
    grid = (N,)
    BLOCK_SIZE = 32  # Process all 32 features at once
    
    # Calculate strides
    s1, s2, s3, s4 = in_1.stride()
    s1_in2, s2_in2, s3_in2, s4_in2 = in_2.stride()
    s1_in3, s2_in3 = in_3.stride()
    s1_out, s2_out, s3_out = out.stride()
    
    # Extract relevant strides for batch and feature dimensions
    stride_in1_batch = s2
    stride_in1_feat = s3  
    stride_in1_inner = s4
    
    stride_in2_batch = s2_in2
    stride_in2_feat = s3_in2
    stride_in2_inner = s4_in2
    
    stride_in3_batch = s1_in3
    stride_in3_feat = s2_in3
    
    stride_out_batch = s2_out
    stride_out_feat = s3_out
    
    attention_kernel[grid](
        in_1,
        in_2,
        in_3,
        out,
        stride_in1_batch,
        stride_in1_feat,
        stride_in1_inner,
        stride_in2_batch,
        stride_in2_feat,
        stride_in2_inner,
        stride_in3_batch,
        stride_in3_feat,
        stride_out_batch,
        stride_out_feat,
        N, K, D,
        BLOCK_SIZE
    )
    
    return out

def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim=2)
    return tmp_5

def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)

def replacement_func():
    return fused_attention_computation