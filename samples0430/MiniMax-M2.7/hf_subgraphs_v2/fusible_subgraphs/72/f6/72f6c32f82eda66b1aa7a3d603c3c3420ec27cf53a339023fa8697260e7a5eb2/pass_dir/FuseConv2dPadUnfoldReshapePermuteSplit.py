import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_pad_unfold_reshape_kernel(
    # Input pointers
    input_ptr, weight_ptr,
    # Output pointers
    out_k_ptr, out_v_ptr,
    # Strides
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_out_channel_stride, weight_in_channel_stride, weight_h_stride, weight_w_stride,
    out_k_batch_stride, out_k_head_stride, out_k_seq_stride, out_k_dim_stride,
    out_v_batch_stride, out_v_head_stride, out_v_seq_stride, out_v_dim_stride,
    # Shapes
    batch_size, in_channels, out_channels, height, width,
    unfold_h, unfold_w, window_size, stride,
    # Split dimension
    split_dim,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    """Fused kernel for: conv2d -> pad -> unfold -> reshape -> permute -> split -> transpose"""
    
    # Get program ID
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Calculate how many windows we need to process
    num_windows_h = (height + stride - 1) // stride
    num_windows_w = (width + stride - 1) // stride
    
    # Calculate output dimensions
    # After unfold: (batch, channels, H_win, W_win, window, window)
    # After reshape to (8, out_channels/8, 4, 9): head_dim = 4, seq_len = 9
    num_heads = 8
    head_dim = 4
    seq_len = 9  # 3 * 3 windows
    query_dim = split_dim
    
    # Each program handles one output element in the K/V output
    # K shape: [batch, num_heads, seq_len, query_dim]
    # V shape: [batch, num_heads, seq_len, key_dim]
    
    # Calculate which output element this program computes
    h_idx = pid_h % seq_len
    w_idx = pid_w % head_dim
    head_idx = pid_h // seq_len
    batch_idx = pid_batch
    
    # Compute output indices
    out_k_idx = batch_idx * out_k_batch_stride + head_idx * out_k_head_stride + h_idx * out_k_seq_stride + w_idx * out_k_dim_stride
    out_v_idx = batch_idx * out_v_batch_stride + head_idx * out_v_head_stride + h_idx * out_v_seq_stride + (w_idx + query_dim) * out_v_dim_stride
    
    # Accumulator for convolution
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Compute which input window this output comes from
    # After unfold, each output element maps to a position in the unfolded tensor
    # The reshape operation reorganizes: [batch, channels, H_win, W_win, win_h, win_w] -> [8, channels/8, 4, 9, window, window]
    # where 4 = 2*2 (2x2 windows per spatial position) and 9 = 3*3 (3x3 spatial positions)
    
    # For head_idx, we select which output channel group (out_channels / 8)
    # For h_idx (0-8), we select: spatial_pos = h_idx // 3, inner_win = h_idx % 3
    spatial_pos_h = h_idx // 3
    spatial_pos_w = w_idx // 2
    inner_pos = (h_idx % 3) * 3 + (w_idx % 2)
    
    # Calculate the starting position in the padded input
    pad_start_h = spatial_pos_h * stride
    pad_start_w = spatial_pos_w * stride
    
    # Offsets for the unfold window
    off_h = pid_h % window_size
    off_w = pid_w % window_size
    
    # Iterate over input channels (for convolution)
    for c in range(0, in_channels, BLOCK_SIZE_H):
        # Load input patch
        in_h = pad_start_h + off_h
        in_w = pad_start_w + off_w
        
        # Bounds check (pad adds 2 on each side)
        # Original: [1, in_channels, 16, 16], after pad: [1, in_channels, 20, 20]
        # Pad is [2, 2, 2, 2] meaning add 2 on each side
        if in_h >= 0 and in_h < height + 4 and in_w >= 0 and in_w < width + 4:
            # Adjust for padding (input is actually in range [2, height+2])
            actual_h = in_h - 2 if in_h >= 2 else 0
            actual_w = in_w - 2 if in_w >= 2 else 0
            
            # Only load if within original bounds
            if actual_h < height and actual_w < width:
                for ci in range(tl.minimum(BLOCK_SIZE_H, in_channels - c)):
                    in_offset = (batch_idx * input_batch_stride + 
                                (c + ci) * input_channel_stride + 
                                actual_h * input_h_stride + 
                                actual_w * input_w_stride)
                    in_val = tl.load(input_ptr + in_offset)
                    
                    # Load weight for this output channel group
                    # Output channel = head_idx * (out_channels/8) + (c + ci) / 8
                    out_ch_group = (c + ci) // 8
                    actual_out_ch = head_idx * (out_channels // 8) + out_ch_group
                    weight_offset = (actual_out_ch * weight_out_channel_stride + 
                                    (c + ci) * weight_in_channel_stride +
                                    0 * weight_h_stride + 0 * weight_w_stride)
                    w_val = tl.load(weight_ptr + weight_offset)
                    
                    acc[ci, 0] += in_val * w_val
    
    # Reduce across channels (sum over input channels / 8 groups)
    result = tl.sum(acc, axis=0)
    
    # Determine which output this goes to (K or V based on split dimension)
    if w_idx < split_dim:
        tl.store(out_k_ptr + out_k_idx, result.to(tl.float16))
    else:
        tl.store(out_v_ptr + out_v_idx, result.to(tl.float16))


@torch.fx.wrap
def fused_conv_pad_unfold_reshape_kernel_wrapper(in_0, in_1, split_dim=16):
    """
    Fused kernel that performs:
    conv2d -> pad(2,2,2,2) -> unfold(2,12,8) -> unfold(3,12,8) -> reshape -> permute -> split -> transpose
    
    Args:
        in_0: weight tensor [out_channels, in_channels, 1, 1]
        in_1: input tensor [batch, in_channels, H, W]
        split_dim: dimension to split the output (default 16 for K, remaining for V)
    
    Returns:
        K: [batch, 8, 9, split_dim] - transposed
        V: [batch, 8, 9, out_channels - split_dim]
    """
    batch, in_channels, height, width = in_1.shape
    out_channels = in_0.shape[0]
    
    # Strides for input
    input_batch_stride = in_1.stride(0)
    input_channel_stride = in_1.stride(1)
    input_h_stride = in_1.stride(2)
    input_w_stride = in_1.stride(3)
    
    # Strides for weight (conv2d)
    weight_out_channel_stride = in_0.stride(0)
    weight_in_channel_stride = in_0.stride(1)
    weight_h_stride = in_0.stride(2)
    weight_w_stride = in_0.stride(3)
    
    # Output dimensions after the full computation
    num_heads = 8
    head_dim = 4
    seq_len = 9  # 3x3 windows
    key_dim = out_channels // 8 - split_dim
    
    # Allocate output tensors
    # K: [batch, num_heads, seq_len, split_dim]
    # V: [batch, num_heads, seq_len, key_dim]
    out_k = torch.empty((batch, num_heads, seq_len, split_dim), 
                        dtype=in_1.dtype, device=in_1.device)
    out_v = torch.empty((batch, num_heads, seq_len, key_dim), 
                        dtype=in_1.dtype, device=in_1.device)
    
    # Strides for outputs
    out_k_batch_stride = out_k.stride(0)
    out_k_head_stride = out_k.stride(1)
    out_k_seq_stride = out_k.stride(2)
    out_k_dim_stride = out_k.stride(3)
    
    out_v_batch_stride = out_v.stride(0)
    out_v_head_stride = out_v.stride(1)
    out_v_seq_stride = out_v.stride(2)
    out_v_dim_stride = out_v.stride(3)
    
    # Grid configuration
    # We process (seq_len * num_heads, head_dim) output elements
    grid = (seq_len, head_dim, batch)
    
    # Launch kernel
    fused_conv_pad_unfold_reshape_kernel[grid](
        in_1, in_0,
        out_k, out_v,
        input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
        weight_out_channel_stride, weight_in_channel_stride, weight_h_stride, weight_w_stride,
        out_k_batch_stride, out_k_head_stride, out_k_seq_stride, out_k_dim_stride,
        out_v_batch_stride, out_v_head_stride, out_v_seq_stride, out_v_dim_stride,
        batch, in_channels, out_channels, height, width,
        3, 3, 12, 8,  # unfold params
        split_dim,
        BLOCK_SIZE_H=32, BLOCK_SIZE_W=1,
    )
    
    return out_k, out_v


def pattern(in_0, in_1):
    """Match the full computation pattern for eca_halonext26ts attention computation"""
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 64], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return tmp_10, tmp_9


def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement - split_dim is 16 for this pattern"""
    return (in_0, in_1, 16)


def replacement_func():
    """Return the replacement function"""
    return fused_conv_pad_unfold_reshape_kernel_wrapper