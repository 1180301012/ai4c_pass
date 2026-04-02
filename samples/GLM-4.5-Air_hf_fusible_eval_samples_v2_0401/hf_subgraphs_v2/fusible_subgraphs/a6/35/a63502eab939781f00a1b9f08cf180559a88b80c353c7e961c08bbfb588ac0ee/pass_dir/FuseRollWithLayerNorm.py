import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact dataflow in model.py
def pattern(in_0, in_1, in_2, in_3):
    """Pattern matches: contiguous -> view -> roll -> view -> layer_norm -> add"""
    tmp_2 = in_3.contiguous()
    # Use flexible dimensions to match both patterns
    if in_2.shape[-1] == 768:
        tmp_3 = tmp_2.view(-1, 32, 32, 768)  # H=32, W=32, C=768 for first case
        tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
        tmp_5 = tmp_4.view(1, 1024, 768)  # N=1024
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    else:  # in_2.shape[-1] == 384
        tmp_3 = tmp_2.view(-1, 64, 64, 384)  # H=64, W=64, C=384 for second case  
        tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
        tmp_5 = tmp_4.view(1, 4096, 384)  # N=4096
        tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel for fused roll operation with layer normalization
@triton.jit
def fused_roll_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    spatial_h,
    spatial_w,
    roll_shift_h,
    roll_shift_w,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ids for 3D launch: [batch, seq_len_blocks, hidden_blocks]
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Compute offsets
    batch_offset = pid_batch * seq_len * hidden_dim
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create bounds checking masks
    m_mask = m_offset + tl.arange(0, BLOCK_SIZE_M) < seq_len
    n_mask = n_offset + tl.arange(0, BLOCK_SIZE_N) < hidden_dim
    mask = m_mask[:, None] & n_mask[None, :]
    
    # Calculate input base address (treating as 1D sequence of length seq_len * hidden_dim)
    input_base = input_ptr + batch_offset + m_offset * hidden_dim
    
    # Load input block
    offsets = n_offset + tl.arange(0, BLOCK_SIZE_N)
    input_block = tl.load(input_base + offsets[None, :], mask=mask, other=0.0)
    
    # Apply roll operation on spatial dimensions
    if roll_shift_h > 0 or roll_shift_w > 0:
        rolled_block = tl.zeros_like(input_block)
        spatial_size = spatial_h * spatial_w
        
        for i in range(BLOCK_SIZE_M):
            if m_mask[i]:
                pos = m_offset + i
                if pos < seq_len:
                    # Convert linear position to 2D spatial coordinates
                    h_pos = pos // spatial_w
                    w_pos = pos % spatial_w
                    
                    # Apply roll shifts and convert back to linear position
                    new_h = (h_pos + roll_shift_h) % spatial_h
                    new_w = (w_pos + roll_shift_w) % spatial_w
                    new_pos = new_h * spatial_w + new_w
                    
                    if new_pos < seq_len:
                        # Load data from rolled position
                        rolled_offset = batch_offset + new_pos * hidden_dim
                        rolled_block[i, :] = tl.load(input_ptr + rolled_offset + offsets, mask=n_mask, other=0.0)
        input_block = rolled_block
    
    # Load weight and bias vectors
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE_N), mask=n_mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE_N), mask=n_mask, other=0.0)
    
    # Load residual
    residual = tl.load(residual_ptr + batch_offset + m_offset * hidden_dim + offsets[None, :], 
                       mask=mask, other=0.0)
    
    # Apply layer normalization
    mean = tl.sum(input_block, axis=1, mask=m_mask) / hidden_dim
    mean = mean[:, None]
    
    centered = input_block - mean
    var = tl.sum(centered * centered, axis=1, mask=m_mask) / hidden_dim
    var = var[:, None]
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    normalized = centered * inv_std
    output = normalized * weight[None, :] + bias[None, :] + residual
    
    # Store result
    output_base = output_ptr + batch_offset + m_offset * hidden_dim
    tl.store(output_base + offsets[None, :], output, mask=mask)

@torch.fx.wrap
def fused_roll_layernorm_forward(in_0, in_1, in_2, in_3):
    # Handle both possible input shapes
    if in_3.dim() == 6:  # Case 1: [1, 4, 8, 4, 8, 768] -> [1, 1024, 768]
        # Reshape from [1, 4, 8, 4, 8, 768] to [-1, 32, 32, 768]
        tmp_2 = in_3.contiguous()
        batch_size_h = in_3.shape[1] * in_3.shape[3]  # 4 * 4 = 16
        batch_size_w = in_3.shape[2] * in_3.shape[4]  # 8 * 8 = 64
        spatial_h = in_3.shape[1] * in_3.shape[3]      # 4 * 4 = 16? Let me recalc...
        # Actually, let's be more careful about the reshaping
        # Pattern: view(-1, 32, 32, 768) from shape [1, 4, 8, 4, 8, 768]
        # This means: -1 = (1 * 4 * 4) = 16, so final view is [16, 32, 32, 768]
        # But wait, that doesn't work with the subsequent view(1, 1024, 768)
        # Let me trace this more carefully:
        # [1, 4, 8, 4, 8, 768] -> view(-1, 32, 32, 768) -> [1, 1024, 768]
        # This implies: -1 * 32 * 32 = 1024, so -1 = 1024 / (32*32) = 1
        # This suggests the view is [1, 32, 32, 768]
        
        # Let me recalculate: 4*8*4*8 = 1024, so -1 = 1024 / (32*32) = 1
        # The view is [1, 32, 32, 768] where 32,32 comes from flattening the intermediate dimensions
        reshaped = tmp_2.view(1, 32, 32, 768)
        seq_len = 1024  # 32 * 32
        hidden_dim = 768
        roll_shift_h = 4
        roll_shift_w = 4
        
    elif in_3.dim() == 6:  # Case 2: [1, 8, 8, 8, 8, 384] -> [1, 4096, 384]
        # Pattern: view(-1, 64, 64, 384) from shape [1, 8, 8, 8, 8, 384]
        # This becomes [1, 64, 64, 384], then view(1, 4096, 384)
        tmp_2 = in_3.contiguous()
        reshaped = tmp_2.view(1, 64, 64, 384)
        seq_len = 4096  # 64 * 64
        hidden_dim = 384
        roll_shift_h = 4
        roll_shift_w = 4
    
    else:
        raise ValueError(f"Unexpected input shape: {in_3.shape}")
    
    # Get input tensor shapes
    batch_size = 1
    input_data = reshaped  # This is now [1, H, W, C]
    
    # Create output tensor
    output_shape = (1, seq_len, hidden_dim)
    output = torch.empty(output_shape, dtype=in_3.dtype, device=in_3.device)
    
    # Set up Triton kernel launch configuration
    BLOCK_SIZE_M = 32  # Sequence length block size
    BLOCK_SIZE_N = 128  # Hidden dimension block size
    
    num_blocks_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (hidden_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_roll_layernorm_kernel[(batch_size, num_blocks_m, num_blocks_n)](
        input_data,
        in_1,  # weight
        in_0,  # bias
        in_2,  # residual
        output,
        batch_size,
        seq_len,
        hidden_dim,
        roll_shift_h,
        roll_shift_w,
        1e-05,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return output.unsqueeze(0) if output.dim() == 2 else output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_roll_layernorm_forward