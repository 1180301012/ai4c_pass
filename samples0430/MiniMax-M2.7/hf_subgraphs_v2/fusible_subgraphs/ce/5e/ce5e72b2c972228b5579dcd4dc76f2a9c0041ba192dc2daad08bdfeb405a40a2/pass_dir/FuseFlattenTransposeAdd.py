import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel_3d(
    conv_out_ptr,
    pos_emb_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    feat_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized kernel for 3D input [1, feat_dim, seq_len] -> [1, seq_len, feat_dim]
    Uses 2D grid for better parallelism.
    """
    # Block row and column indices
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    
    # Starting indices for this block
    row_start = row_pid * BLOCK_SIZE_M
    col_start = col_pid * BLOCK_SIZE_K
    
    # Row and column offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_K)
    
    # Mask for valid elements
    row_mask = row_offsets < seq_len
    col_mask = col_offsets < feat_dim
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load convolution output: [0, c, s] where c = col_offset, s = row_offset
    # Contiguous layout: flat_idx = c * seq_len + s
    conv_offsets = col_offsets[None, :] * seq_len + row_offsets[:, None]
    conv_vals = tl.load(conv_out_ptr + conv_offsets, mask=mask, other=0.0)
    
    # Load position embeddings: [0, s, c] where s = row_offset, c = col_offset
    pos_offsets = row_offsets[:, None] * feat_dim + col_offsets[None, :]
    pos_vals = tl.load(pos_emb_ptr + pos_offsets, mask=mask, other=0.0)
    
    # Compute output: [0, s, c] = conv + pos
    out_vals = conv_vals + pos_vals
    
    # Store to output: [0, s, c]
    out_offsets = row_offsets[:, None] * feat_dim + col_offsets[None, :]
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_impl_3d(conv_out, pos_emb, seq_len, feat_dim):
    """
    Handle 3D input: [1, feat_dim, seq_len]
    Optimized 2D grid launch.
    """
    output_shape = (1, seq_len, feat_dim)
    out = torch.empty(output_shape, device=conv_out.device, dtype=conv_out.dtype)
    
    # 2D grid: [num_row_blocks, num_col_blocks]
    # Each block processes BLOCK_SIZE_M rows and BLOCK_SIZE_K columns
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_K = 64
    
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (feat_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (grid_m, grid_k)
    
    fused_kernel_3d[grid](
        conv_out,
        pos_emb,
        out,
        seq_len,
        feat_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
    )
    
    return out


@triton.jit
def fused_kernel_5d(
    conv_out_ptr,
    pos_emb_ptr,
    out_ptr,
    channels: tl.constexpr,
    d_dim: tl.constexpr,
    h_dim: tl.constexpr,
    w_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized kernel for 5D conv output: [1, channels, D, H, W]
    Uses 2D grid and efficient indexing.
    """
    seq_len = d_dim * h_dim * w_dim
    
    # 2D grid
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    
    # Starting indices
    row_start = row_pid * BLOCK_SIZE_M
    col_start = col_pid * BLOCK_SIZE_K
    
    # Offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_K)
    
    # Mask
    row_mask = row_offsets < seq_len
    col_mask = col_offsets < channels
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load conv output in transposed view
    # conv_out[c, s] where s is the flattened position index
    conv_offsets = col_offsets[None, :] * seq_len + row_offsets[:, None]
    conv_vals = tl.load(conv_out_ptr + conv_offsets, mask=mask, other=0.0)
    
    # Load position embeddings [s, c]
    pos_offsets = row_offsets[:, None] * channels + col_offsets[None, :]
    pos_vals = tl.load(pos_emb_ptr + pos_offsets, mask=mask, other=0.0)
    
    # Add
    out_vals = conv_vals + pos_vals
    
    # Store output [s, c]
    out_offsets = row_offsets[:, None] * channels + col_offsets[None, :]
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_impl_5d(conv_out, pos_emb, channels, d_dim, h_dim, w_dim):
    """
    Handle 5D conv output: [1, channels, D, H, W]
    """
    seq_len = d_dim * h_dim * w_dim
    
    output_shape = (1, seq_len, channels)
    out = torch.empty(output_shape, device=conv_out.device, dtype=conv_out.dtype)
    
    # 2D grid
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_K = 64
    
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (grid_m, grid_k)
    
    fused_kernel_5d[grid](
        conv_out,
        pos_emb,
        out,
        channels,
        d_dim,
        h_dim,
        w_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
    )
    
    return out


def replacement_impl(a, b):
    """
    Module-level replacement function.
    
    The framework passes conv3d output and position embeddings.
    We need to handle reshape/transpose/add in the Triton kernel.
    """
    # Only tensor allocation APIs allowed here!
    
    if len(a.shape) == 5:
        # a is [1, channels, D, H, W]
        channels = a.shape[1]
        d_dim = a.shape[2]
        h_dim = a.shape[3]
        w_dim = a.shape[4]
        return fused_impl_5d(a, b, channels, d_dim, h_dim, w_dim)
    elif len(a.shape) == 3:
        # a is [1, channels, seq_len]
        seq_len = a.shape[2]
        feat_dim = a.shape[1]
        return fused_impl_3d(a, b, seq_len, feat_dim)
    else:
        # Fallback
        return a + b


def pattern(a, b):
    """
    Match pattern:
    tmp_4 = a.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = b.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device = device(type='cuda', index=0), copy = True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9
    """
    tmp_4 = a.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = b.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=torch.device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(a, b):
    """
    Extract arguments needed for the fused kernel.
    """
    return (a, b)


def replacement_func():
    """
    Return the replacement function - must return a module-level function.
    """
    return replacement_impl