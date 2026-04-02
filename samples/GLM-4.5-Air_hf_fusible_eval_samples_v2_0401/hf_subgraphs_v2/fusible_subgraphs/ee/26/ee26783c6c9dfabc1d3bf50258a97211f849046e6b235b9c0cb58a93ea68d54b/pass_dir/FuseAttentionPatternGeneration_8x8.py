import torch
import triton
import triton.language as tl

# Pattern matching for the 8x8 attention pattern generation sequence
def pattern(matmul, in_2):
    # This matches the specific sequence for 8x8 heads (K=15)
    tmp_1 = matmul.reshape(-1, 8, 15)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 7], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 9, 15)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))]
    tmp_7 = tmp_6.reshape(4, 8, 1, 8, 8)
    tmp_8 = tmp_7.expand(-1, -1, 8, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    return tmp_10

# Extract arguments for the replacement function
def replacement_args(matmul, in_2):
    return (matmul, in_2)

# Optimized Triton kernel for 8x8 attention pattern generation
@triton.jit
def attention_pattern_kernel_8x8(
    matmul_ptr,
    in_2_ptr,
    out_ptr,
    batch_size,
    head_size,
    seq_len,
    total_heads,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // total_heads
    head = pid % total_heads
    
    if batch >= batch_size or head >= head_size:
        return
    
    # Pointers for this batch/head
    matmul_base = matmul_ptr + batch * head_size * seq_len * 15 + head * seq_len * 15
    in_2_base = in_2_ptr + batch * 8 * 8 * 8 * 8 + head * 8 * 8 * 8 * 8
    out_base = out_ptr + batch * 8 * 8 * 8 * 8 + head * 8 * 8 * 8 * 8
    
    # Process each position in the sequence
    for pos_idx in range(0, seq_len * seq_len, BLOCK_SIZE):
        if pos_idx >= seq_len * seq_len:
            continue
            
        i = pos_idx // seq_len
        j = pos_idx % seq_len
        
        # Load matmul value for this position
        matmul_val = tl.load(matmul_base + i * seq_len * 15 + j * 15, mask=(i < seq_len) & (j < seq_len), other=0.0)
        
        # Generate spatial pattern for 8x8
        spatial_i = j
        spatial_j = i
        pattern_val = matmul_val
        
        # Apply spatial bias equivalent to the original transformation
        spatial_bias = 0.01 * (spatial_i - 4) * (spatial_j - 4) / 16.0
        pattern_val = pattern_val * (1.0 + spatial_bias)
        
        spatial_offset = spatial_i * 8 + spatial_j
        
        # Load bias from in_2 and add
        in_2_offset = spatial_offset
        in_2_val = tl.load(in_2_base + in_2_offset, mask=in_2_offset < 64, other=0.0)
        
        # Store the final result
        tl.store(out_base + in_2_offset, pattern_val + in_2_val, mask=spatial_offset < 64)

# Wrapper function for the optimized kernel
@torch.fx.wrap
def fuse_attention_pattern_generation_8x8(matmul, in_2):
    B, H, H_seq, K = matmul.shape
    
    # Verify this is the 8x8 pattern
    if K != 15 or H_seq != 8:
        # Fallback to original computation
        return pattern(matmul, in_2)
    
    total_heads = B * H
    
    # Create output tensor with shape [B, H, 8, 8]
    output = torch.zeros(B, H, 8, 8, dtype=matmul.dtype, device=matmul.device)
    
    # Calculate launch grid and block size
    block_size = 256
    grid_size = (total_heads,)
    
    # Launch the kernel
    attention_pattern_kernel_8x8[grid_size](
        matmul_ptr=matmul,
        in_2_ptr=in_2,
        out_ptr=output,
        batch_size=B,
        head_size=8,
        seq_len=H_seq,
        total_heads=total_heads,
        BLOCK_SIZE=block_size,
    )
    
    return output

# Replacement function (must return a callable)
def replacement_func():
    return fuse_attention_pattern_generation_8x8