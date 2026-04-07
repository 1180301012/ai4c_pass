import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_unfold_kernel(
    x_ptr,
    out_ptr,
    in_channels,
    seq_len,
    out_channels,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offset = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = n_offset < (seq_len - 8)  # Account for kernel size with padding
    
    # Process each channel
    for c in range(0, in_channels, BLOCK_SIZE_C):
        c_block = tl.arange(0, BLOCK_SIZE_C)
        mask_c = c_block < (in_channels - c)
        
        # Load input chunk
        x = tl.load(x_ptr + c * seq_len + n_offset[:, None], mask=(mask_n[:, None] & mask_c[None, :]), other=0.0)
        
        # Apply unfold manually with convolution-style sliding window
        for i in range(9):
            n_start = n_offset + i
            mask_window = n_start < seq_len
            # Extract sliding window
            window = x[:, n_start - 4]  # Account for 4 padding on left
            # Store to output with proper indexing
            out_idx = (c_block[:, None] * 9 + i) * (seq_len - 8) + (n_start - 4)
            tl.store(out_ptr + out_idx, window, mask=(mask_window[None, :] & mask_c[:, None]))
    
    # Second phase: transpose and reshape
    total_patches = in_channels * (seq_len - 8)
    for idx in range(0, total_patches, BLOCK_SIZE_N):
        idx_block = idx + tl.arange(0, BLOCK_SIZE_N)
        mask_idx = idx_block < total_patches
        
        # Load current data
        current = tl.load(out_ptr + idx_block, mask=mask_idx, other=0.0)
        
        # Transform to final output shape: (N, C_out, 9)
        final_idx = ((idx_block // (seq_len - 8)) // 6) * 64 * 9 + (idx_block % (seq_len - 8)) * 9 + tl.arange(0, 9) % 9
        mask_final = final_idx < (out_channels * (seq_len - 8 - 1) * 9)
        
        tl.store(out_ptr + final_idx, current, mask=mask_final)

@torch.fx.wrap
def optimized_unfold_reshape(x, in_channels, seq_len):
    in_3d = x.unsqueeze(-1)  # Add channel dim for computation
    batch_size, in_c, in_s = in_3d.shape
    
    # Calculate output dimensions
    out_patches = in_c * (in_s - 8)  # 9 kernel size with stride 1
    out_channels = 64  # Based on the reshape to [-1, 64, 9]
    total_elements = out_patches * 9
    
    # Output shape: [batch_size, seq_len-8, in_c, 9] -> [batch_size*(seq_len-8)*64, 9] wait, let me recalculate
    
    # From the pattern: reshape(1, -1, 384, 9) then torch.reshape([-1, 64, 9])
    # -1 becomes batch_size * seq_len_patches, and we get [batch_size, seq_len_patches, in_c//6, 9]
    # where 384 = 64 * 6, so in_c must be 384, out_c = 64
    
    assert in_channels == 384, f"Expected in_channels=384, got {in_channels}"
    seq_patches = seq_len - 8
    
    total_elements = batch_size * seq_patches * 64 * 9
    out = torch.empty((batch_size, seq_patches, 64, 9), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    BLOCK_SIZE_C = 64
    BLOCK_SIZE_N = 128
    total_patches = in_channels * seq_patches
    
    grid = (total_patches + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_unfold_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        in_channels=in_channels,
        seq_len=seq_len,
        out_channels=out_channels,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out.reshape(-1, 64, 9)

def replacement_func():
    # Return a function that can be called with input
    def optimized_func(x):
        in_channels = x.shape[1]
        seq_len = x.shape[2]
        return optimized_unfold_reshape(x, in_channels, seq_len)
    return optimized_func