import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel(
    # Input pointers
    input_ptr, weight_ptr,
    # Output pointers
    out_k_ptr, out_v_ptr,
    # Input strides
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    # Weight strides
    weight_out_ch_stride, weight_in_ch_stride,
    # Output K strides
    out_k_batch_stride, out_k_head_stride, out_k_seq_stride, out_k_dim_stride,
    # Output V strides  
    out_v_batch_stride, out_v_head_stride, out_v_seq_stride, out_v_dim_stride,
    # Shapes
    batch, in_channels, out_channels, height, width,
    # Attention params
    split_dim, num_heads, head_dim, seq_len, stride, window_size,
    BLOCK_SIZE: tl.constexpr,
    out_dtype: tl.constexpr,
):
    """Fused kernel for the full attention computation: conv2d -> pad -> unfold -> reshape -> permute -> split -> transpose
    
    This kernel avoids materializing intermediate tensors by computing everything in one pass.
    """
    # Grid: (head_dim * seq_len, batch)
    # Each program computes one (head, seq_pos) combination for all batch elements
    pid_h = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    # Calculate which head and sequence position
    h_idx = pid_h % seq_len
    w_idx = (pid_h // seq_len) % head_dim
    head_idx = pid_h // (seq_len * head_dim)
    
    # Spatial position in the unfolded grid (3x3 = 9 positions)
    spatial_h = h_idx // 3
    spatial_w = h_idx % 3
    
    # Compute starting positions in padded input
    pad_start_h = spatial_h * stride
    pad_start_w = spatial_w * stride
    
    # Accumulator
    acc_dtype = tl.float32
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    
    # Iterate over input channels
    for c in range(0, in_channels, BLOCK_SIZE):
        offs = tl.arange(0, BLOCK_SIZE) + c
        c_offs = offs.to(tl.int64)
        mask = offs < in_channels
        
        # Calculate input positions for this window
        # The unfold creates sliding windows, we need position (w_idx % 2, w_idx // 2) in the 12x12 window
        win_h = (w_idx // 2) % 3
        win_w = (w_idx % 2) + (w_idx // 4) * 2
        
        in_h = pad_start_h + win_h
        in_w = pad_start_w + win_w
        
        # Bounds check (with padding)
        h_mask = (in_h >= 0) & (in_h < height + 4)
        w_mask = (in_w >= 0) & (in_w < width + 4)
        bounds_mask = h_mask & w_mask & mask
        
        # Convert padded coords to original coords
        orig_h = tl.where(in_h >= 2, in_h - 2, 0)
        orig_w = tl.where(in_w >= 2, in_w - 2, 0)
        
        # Load input
        in_offsets = pid_batch * input_batch_stride + c_offs * input_channel_stride + orig_h * input_h_stride + orig_w * input_w_stride
        in_ptrs = input_ptr + in_offsets
        in_vals = tl.load(in_ptrs, mask=bounds_mask, other=0.0).to(acc_dtype)
        
        # Load weight: out_ch = head_idx * (out_channels/8) + c/8
        out_ch_group = (offs // 8).to(tl.int64)
        out_ch = head_idx * (out_channels // num_heads) + out_ch_group
        w_offsets = out_ch * weight_out_ch_stride + c_offs * weight_in_ch_stride
        w_ptrs = weight_ptr + w_offsets
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0).to(acc_dtype)
        
        acc = acc + in_vals * w_vals
    
    result = tl.sum(acc, axis=0)
    
    # Compute output offsets
    out_k_idx = (pid_batch * out_k_batch_stride + head_idx * out_k_head_stride + h_idx * out_k_seq_stride + w_idx * out_k_dim_stride)
    out_v_idx = (pid_batch * out_v_batch_stride + head_idx * out_v_head_stride + h_idx * out_v_seq_stride + (w_idx + split_dim) * out_v_dim_stride)
    
    # Store to outputs
    tl.store(out_k_ptr + out_k_idx, result.to(out_dtype))
    tl.store(out_v_ptr + out_v_idx, result.to(out_dtype))


@torch.fx.wrap
def fused_attention_wrapper(in_0, in_1, split_dim=16):
    """Fused attention computation: conv2d + pad + unfold + reshape + permute + split + transpose"""
    batch, in_channels, height, width = in_1.shape
    out_channels = in_0.shape[0]
    
    # Attention parameters
    num_heads = 8
    head_dim = 4
    seq_len = 9  # 3x3 windows
    stride = 8
    window_size = 12
    
    key_dim = out_channels // num_heads - split_dim
    
    dtype = in_1.dtype
    
    # Allocate outputs
    # K: [batch, num_heads, seq_len, split_dim]
    # V: [batch, num_heads, seq_len, key_dim]
    out_k = torch.empty((batch, num_heads, seq_len, split_dim), dtype=dtype, device=in_1.device)
    out_v = torch.empty((batch, num_heads, seq_len, key_dim), dtype=dtype, device=in_1.device)
    
    # Convert dtype to Triton type
    if dtype == torch.float32:
        out_dtype = tl.float32
    elif dtype == torch.float16:
        out_dtype = tl.float16
    elif dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float16
    
    # Grid: (head_dim * seq_len, batch)
    grid = (head_dim * seq_len * num_heads, batch)
    
    fused_attention_kernel[grid](
        in_1, in_0,
        out_k, out_v,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(1),
        out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
        out_v.stride(0), out_v.stride(1), out_v.stride(2), out_v.stride(3),
        batch, in_channels, out_channels, height, width,
        split_dim, num_heads, head_dim, seq_len, stride, window_size,
        BLOCK_SIZE=64,
        out_dtype=out_dtype,
    )
    
    return out_k, out_v


def pattern(in_0, in_1):
    """Match the full computation pattern for 384 channel models (bfloat16/float32)"""
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 48, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 32], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return tmp_10, tmp_9


def replacement_args(in_0, in_1):
    """Extract arguments - split_dim is 16 for this pattern"""
    return (in_0, in_1, 16)


def replacement_func():
    """Return the replacement function"""
    return fused_attention_wrapper