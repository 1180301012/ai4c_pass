import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_pad_unfold_384_kernel(
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
    # Split dimension
    split_dim,
):
    """Fused kernel for: conv2d -> pad -> unfold -> reshape -> permute -> split -> transpose
    Optimized for 384 output channels (48 * 8 = 384)
    """
    
    # Get program ID
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Output dimensions
    num_heads = 8
    head_dim = 4
    seq_len = 9  # 3x3 windows
    stride = 8
    window_size = 12
    
    # Calculate which output element this program computes
    h_idx = pid_h % seq_len
    w_idx = pid_w % head_dim
    head_idx = pid_h // seq_len
    batch_idx = pid_batch
    
    # Compute output indices
    out_k_idx = batch_idx * out_k_batch_stride + head_idx * out_k_head_stride + h_idx * out_k_seq_stride + w_idx * out_k_dim_stride
    out_v_idx = batch_idx * out_v_batch_stride + head_idx * out_v_head_stride + h_idx * out_v_seq_stride + (w_idx + split_dim) * out_v_dim_stride
    
    # Compute which input window this output comes from
    spatial_pos_h = h_idx // 3
    spatial_pos_w = w_idx // 2
    
    # Calculate the starting position in the padded input
    pad_start_h = spatial_pos_h * stride
    pad_start_w = spatial_pos_w * stride
    
    # Offsets for the unfold window
    off_h = h_idx % 3
    off_w = w_idx % 2
    
    # Accumulator for convolution
    acc = tl.zeros((), dtype=tl.float32)
    
    # Iterate over input channels (for convolution)
    for c in range(in_channels):
        # Calculate input position
        in_h = pad_start_h + off_h
        in_w = pad_start_w + off_w
        
        # Bounds check (pad adds 2 on each side)
        if in_h >= 2 and in_h < height + 2 and in_w >= 2 and in_w < width + 2:
            actual_h = in_h - 2
            actual_w = in_w - 2
            
            # Load input value
            in_offset = (batch_idx * input_batch_stride + 
                        c * input_channel_stride + 
                        actual_h * input_h_stride + 
                        actual_w * input_w_stride)
            in_val = tl.load(input_ptr + in_offset)
            
            # Load weight: out_ch = head_idx * (out_channels/8) + c/8
            out_ch_group = c // 8
            actual_out_ch = head_idx * (out_channels // 8) + out_ch_group
            weight_offset = (actual_out_ch * weight_out_channel_stride + 
                            c * weight_in_channel_stride)
            w_val = tl.load(weight_ptr + weight_offset)
            
            acc += in_val * w_val
    
    # Store result to appropriate output
    if w_idx < split_dim:
        tl.store(out_k_ptr + out_k_idx, acc.to(tl.bfloat16))
    else:
        tl.store(out_v_ptr + out_v_idx, acc.to(tl.bfloat16))


@torch.fx.wrap
def fused_conv_pad_unfold_384_wrapper(in_0, in_1, split_dim=16):
    """
    Fused kernel for 384 output channels pattern:
    conv2d -> pad(2,2,2,2) -> unfold(2,12,8) -> unfold(3,12,8) -> reshape -> permute -> split -> transpose
    
    Args:
        in_0: weight tensor [384, 256, 1, 1]
        in_1: input tensor [1, 256, 16, 16]
        split_dim: dimension to split (16)
    
    Returns:
        K: [1, 8, 9, 16] - transposed
        V: [1, 8, 9, 32]
    """
    batch, in_channels, height, width = in_1.shape
    out_channels = in_0.shape[0]
    
    # Strides for input
    input_batch_stride = in_1.stride(0)
    input_channel_stride = in_1.stride(1)
    input_h_stride = in_1.stride(2)
    input_w_stride = in_1.stride(3)
    
    # Strides for weight
    weight_out_channel_stride = in_0.stride(0)
    weight_in_channel_stride = in_0.stride(1)
    weight_h_stride = in_0.stride(2)
    weight_w_stride = in_0.stride(3)
    
    # Output dimensions
    num_heads = 8
    head_dim = 4
    seq_len = 9
    key_dim = out_channels // 8 - split_dim
    
    # Allocate output tensors
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
    grid = (seq_len, head_dim, batch)
    
    fused_conv_pad_unfold_384_kernel[grid](
        in_1, in_0,
        out_k, out_v,
        input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
        weight_out_channel_stride, weight_in_channel_stride, weight_h_stride, weight_w_stride,
        out_k_batch_stride, out_k_head_stride, out_k_seq_stride, out_k_dim_stride,
        out_v_batch_stride, out_v_head_stride, out_v_seq_stride, out_v_dim_stride,
        batch, in_channels, out_channels, height, width,
        split_dim,
    )
    
    return out_k, out_v


def pattern(in_0, in_1):
    """Match the 384 output channels pattern for bfloat16 and float32"""
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
    return fused_conv_pad_unfold_384_wrapper