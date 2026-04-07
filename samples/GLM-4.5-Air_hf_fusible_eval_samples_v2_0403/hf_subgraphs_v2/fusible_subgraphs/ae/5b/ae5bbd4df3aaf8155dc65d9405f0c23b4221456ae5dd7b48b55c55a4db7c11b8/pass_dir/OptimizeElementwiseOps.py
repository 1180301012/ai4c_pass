import torch
import triton
import triton.language as tl

def pattern(tmp_8, tmp_9, in_2):
    """Fuse element-wise operations: multiply, subtract, multiply, add"""
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, tmp_13.shape[1], -1, 1)
    return tmp_14

def replacement_args(tmp_8, tmp_9, in_2):
    return (tmp_8, tmp_9, in_2)

@triton.jit
def fused_elementwise_ops_kernel(
    tmp_8_ptr, tmp_9_ptr, in_2_ptr, out_ptr,
    batch_size, seq_len, n_features,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused elementwise operations kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * seq_len * n_features)
    
    # Load inputs
    offsets_3d = offsets.reshape(-1, 1)
    batch_seq_feat_idx = offsets // (seq_len * n_features)
    within_seq_feat_idx = offsets % (seq_len * n_features)
    seq_idx = within_seq_feat_idx // n_features
    feat_idx = within_seq_feat_idx % n_features
    
    # Load tmp_8 and tmp_9 (both have shape [batch*seq, features])
    tmp_8_val = tl.load(tmp_8_ptr + offsets, mask=mask)
    tmp_9_val = tl.load(tmp_9_ptr + offsets, mask=mask)
    
    # Load in_2 (shape [1, seq_len, 1, 1] -> reshape to [seq_len, 1])
    in_2_val = tl.load(in_2_ptr + seq_idx, mask=seq_idx < seq_len)
    
    # Fused elementwise operations
    tmp_10 = tmp_9_val * in_2_val        # tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0               # tmp_10 - 1.0
    tmp_12 = tmp_8_val * tmp_11         # tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0               # tmp_12 + 2.0
    
    # Store result
    tl.store(out_ptr + offsets, tmp_13, mask=mask)

@torch.fx.wrap
def fused_elementwise_ops(tmp_8, tmp_9, in_2):
    """Wrapper for fused elementwise operations"""
    # Input shapes: tmp_8 and tmp_9 are [batch*seq, features], in_2 is [seq_len, 1]
    output_shape = tmp_8.shape
    batch_size = output_shape[0] // tmp_8.shape[1] if len(tmp_8.shape) >= 2 else 1
    seq_len = tmp_8.shape[1] if len(tmp_8.shape) >= 2 else 1
    n_features = tmp_8.shape[0] // (batch_size * seq_len) if len(tmp_8.shape) >= 2 else 1
    
    # Allocate output tensor
    out = torch.zeros_like(tmp_8)
    
    # Set kernel configuration
    BLOCK_SIZE = 1024
    total_elements = batch_size * seq_len * n_features
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_elementwise_ops_kernel[num_programs](
        tmp_8_ptr=tmp_8,
        tmp_9_ptr=tmp_9,
        in_2_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final reshape to match expected output
    final_out = out.view(1, out.shape[1], -1, 1)
    return final_out

def replacement_func():
    return fused_elementwise_ops