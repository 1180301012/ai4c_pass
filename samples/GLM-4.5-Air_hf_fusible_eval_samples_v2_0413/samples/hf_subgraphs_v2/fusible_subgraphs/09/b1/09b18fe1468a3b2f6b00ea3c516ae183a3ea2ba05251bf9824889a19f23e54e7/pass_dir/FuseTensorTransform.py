import torch
import triton
import triton.language as tl

@triton.jit
def tensor_transform_kernel(
    in_0_ptr,       # [32, 512] -> [1, 1, 32, 512]
    in_4_ptr,       # [1, 4096, 512] -> [1, 4096, 32, 512] 
    tmp_5_ptr,      # [1, 4096, 32] -> [1, 4096, 32, 1]
    out_diff_ptr,   # result of expanded in_4 - viewed in_0
    out_tmp9_ptr,   # result of unsqueezed tmp_5
    stride_in0_0, stride_in0_1,
    stride_in4_0, stride_in4_1, stride_in4_2,
    stride_tmp5_0, stride_tmp5_1, stride_tmp5_2,
    stride_out0, stride_out1, stride_out2, stride_out3,
    stride_out9_0, stride_out9_1, stride_out9_2, stride_out9_3,
    N: tl.constexpr,  # 4096
    K: tl.constexpr,  # 32  
    D: tl.constexpr,  # 512
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= N:
        return
        
    # Process all features for this sample at once
    feat_offsets = tl.arange(0, K) * stride_out2
    sample_offset = pid * stride_out1
    
    # Process chunk of features
    for feat_chunk_start in range(0, K, BLOCK_SIZE):
        chunk_end = min(feat_chunk_start + BLOCK_SIZE, K)
        current_size = chunk_end - feat_chunk_start
        feat_offsets_chunk = feat_offsets[feat_chunk_start:feat_chunk_start+current_size]
        
        # Load viewed in_0 data: [1, 1, 32, 512] -> [1, 4096, 32, 512] (broadcasting across batch)
        # tmp_6: [1, 1, 32, 512], so we load [32, 512] and expand across batch
        in0_feat_offsets = feat_offsets_chunk * stride_in0_1
        in0_data = tl.load(in_0_ptr + in0_feat_offsets)
        
        # Load expanded in_4 data: [1, 4096, 32, 512]
        in4_offsets = sample_offset + feat_offsets_chunk[:, None] * stride_out2 + \
                      (tl.arange(0, D)[:, None]) * stride_out3
        in4_data = tl.load(in_4_ptr + in4_offsets)
        
        # Compute subtraction: expanded_in_4 - viewed_in_0
        diff = in4_data - in0_data
        
        # Store difference result
        out_diff_offsets = sample_offset + feat_offsets_chunk[:, None] * stride_out2 + \
                          (tl.arange(0, D)[:, None]) * stride_out3
        tl.store(out_diff_ptr + out_diff_offsets, diff)
        
        # Load and unsqueeze tmp_5: [1, 4096, 32] -> [1, 4096, 32, 1]
        tmp5_data = tl.load(tmp_5_ptr + sample_offset + feat_offsets_chunk * stride_tmp5_2)
        
        # Store unsqueezed result
        out_tmp9_offsets = sample_offset + feat_offsets_chunk[:, None] * stride_out9_2 + \
                          (1 * stride_out9_3)  # Add dummy dimension
        tl.store(out_tmp9_ptr + out_tmp9_offsets, tmp5_data)

@torch.fx.wrap
def fused_tensor_transform(in_0, in_4, tmp_5):
    # Get tensor shapes
    N, K, D = 4096, 32, 512
    
    # Output shapes
    out_diff = torch.empty((1, N, K, D), dtype=in_0.dtype, device=in_0.device)
    out_tmp9 = torch.empty((1, N, K, 1), dtype=tmp_5.dtype, device=tmp_5.device)
    
    # Configure kernel - one program per batch sample
    grid = (N,)
    BLOCK_SIZE = 32  # Process all features at once
    
    # Calculate strides
    s1_in0, s2_in0 = in_0.stride()
    s1_in4, s2_in4, s3_in4 = in_4.stride()
    s1_tmp5, s2_tmp5, s3_tmp5 = tmp_5.stride()
    s1_diff, s2_diff, s3_diff, s4_diff = out_diff.stride()
    s1_tmp9, s2_tmp9, s3_tmp9, s4_tmp9 = out_tmp9.stride()
    
    # Extract relevant strides
    stride_in0_0, stride_in0_1 = s1_in0, s2_in0
    stride_in4_0, stride_in4_1, stride_in4_2 = s1_in4, s2_in4, s3_in4
    stride_tmp5_0, stride_tmp5_1, stride_tmp5_2 = s1_tmp5, s2_tmp5, s3_tmp5
    
    stride_out0, stride_out1, stride_out2, stride_out3 = s1_diff, s2_diff, s3_diff, s4_diff
    stride_out9_0, stride_out9_1, stride_out9_2, stride_out9_3 = s1_tmp9, s2_tmp9, s3_tmp9, s4_tmp9
    
    tensor_transform_kernel[grid](
        in_0,
        in_4,
        tmp_5,
        out_diff,
        out_tmp9,
        stride_in0_0, stride_in0_1,
        stride_in4_0, stride_in4_1, stride_in4_2,
        stride_tmp5_0, stride_tmp5_1, stride_tmp5_2,
        stride_out0, stride_out1, stride_out2, stride_out3,
        stride_out9_0, stride_out9_1, stride_out9_2, stride_out9_3,
        N, K, D,
        BLOCK_SIZE
    )
    
    return out_diff, out_tmp9

def pattern(in_0, in_4, tmp_5):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_9 = tmp_5.unsqueeze(3)
    tmp_10 = tmp_8 - tmp_6
    return tmp_10, tmp_9

def replacement_args(in_0, in_4, tmp_5):
    return (in_0, in_4, tmp_5)

def replacement_func():
    return fused_tensor_transform