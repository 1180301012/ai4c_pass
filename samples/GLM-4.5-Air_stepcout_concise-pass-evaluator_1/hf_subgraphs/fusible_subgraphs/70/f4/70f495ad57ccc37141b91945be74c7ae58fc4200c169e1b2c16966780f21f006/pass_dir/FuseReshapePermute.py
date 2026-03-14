import torch
import triton
import triton.language as tl

def pattern(tmp_4, batch_size):
    # The reshape + permute fusion - matching exact computation order
    # tmp_4 has shape [batch, seq, features] = [batch, 192, features]
    # reshape to [batch, 16, 12, -1] then permute to [batch, features, 16, 12]
    tmp_5 = tmp_4.reshape(batch_size, 16, 12, -1)
    tmp_6 = tmp_5.permute(0, 3, 1, 2)
    return tmp_6

def replacement_args(tmp_4, batch_size):
    return (tmp_4, batch_size)

@triton.jit
def fused_reshape_permute_kernel(
    x_ptr, out_ptr,
    batch_size, seq_len, n_features,
    height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Directly transform [batch, seq, features] to [batch, features, height, width]
    avoiding intermediate memory allocation
    """
    pid_m = tl.program_id(0)  # features dimension
    pid_n = tl.program_id(1)  # (batch, height, width) flattened
    
    # Calculate batch, height, width indices
    batch_idx = pid_n // (height * width)
    hw_idx = pid_n % (height * width)
    height_idx = hw_idx // width
    width_idx = hw_idx % width
    
    mask = (pid_m < n_features) & (batch_idx < batch_size) & (height_idx < height) & (width_idx < width)
    
    if not mask:
        return
    
    # Load from original layout
    src_offset = batch_idx * seq_len * n_features + height_idx * width + width_idx * n_features + pid_m
    x = tl.load(x_ptr + src_offset, mask=mask)
    
    # Store directly to final layout
    dst_offset = batch_idx * n_features * height * width + pid_m * height * width + height_idx * width + width_idx
    tl.store(out_ptr + dst_offset, x, mask=mask)

@torch.fx.wrap
def fused_reshape_permute(tmp_4, batch_size):
    # height=16, width=12 are fixed from the computation pattern
    height, width = 16, 12
    
    n_batch, seq_len, n_features = tmp_4.shape
    # Calculate the inferred feature dimension
    inferred_features = seq_len // (height * width)
    
    # Optimize tile sizes based on problem size
    BLOCK_SIZE_M = min(128, n_features)
    BLOCK_SIZE_N = max(1, (batch_size * height * width + 31) // 32)
    
    grid = (BLOCK_SIZE_M, (batch_size * height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    out = torch.empty(batch_size, n_features, height, width, dtype=tmp_4.dtype, device=tmp_4.device)
    fused_reshape_permute_kernel[grid](
        tmp_4, out,
        n_batch, seq_len, n_features,
        height, width,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return fused_reshape_permute