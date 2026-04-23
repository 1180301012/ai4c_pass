import torch
import triton
import triton.language as tl

# ============================================================================
# Fused Conv1d + GELU + Slice Pass
# ============================================================================
# Input shapes:
#   in_3: [1, 768, 249] - hidden_states
#   in_4: [768, 48, 31] - conv1d weight  
#   in_2: [768] - conv1d bias
#
# Operations: conv1d(in_3, in_4, in_2) -> gelu -> slice[..., :124]
# Output: tmp_7 with shape [1, 48, 124]

@triton.jit
def fused_conv1d_gelu_slice_kernel(
    in_ptr, weight_ptr, bias_ptr,
    out_ptr,
    batch_size, in_channels, out_channels,
    in_len, kernel_len,
    stride, padding, dilation, groups,
    out_len,  # = 124
    BLOCK_SIZE: tl.constexpr,
):
    # Compute position
    pid = tl.program_id(0)
    batch = pid // (out_channels * out_len)
    out_ch = (pid // out_len) % out_channels
    out_pos = pid % out_len
    
    # Input position for conv (with padding)
    in_base = out_pos * stride - padding
    
    # For grouped conv1d: groups=16, each group has out_channels/groups output channels
    # and in_channels/groups input channels
    group_size = out_channels // groups  # 3
    in_group_size = in_channels // groups  # 48
    group_idx = out_ch // group_size
    out_ch_in_group = out_ch % group_size
    
    # Initialize accumulator with bias (cast to fp32 for accumulation)
    acc = tl.cast(tl.load(bias_ptr + out_ch), tl.float32)
    
    # Conv1d accumulation for grouped convolution (accumulate in fp32)
    for k in range(kernel_len):
        in_base_pos = in_base + k * dilation
        # Check bounds: need both conditions
        is_valid = (in_base_pos >= 0) and (in_base_pos < in_len)
        if is_valid:
            # For grouped conv: weight index = group_idx * group_size * in_group_size * kernel_len
            #                 + out_ch_in_group * in_group_size * kernel_len
            #                 + in_ch_in_group * kernel_len + k
            # where in_ch_in_group iterates over input channels within the group
            for in_ch_in_group in range(in_group_size):
                in_ch = group_idx * in_group_size + in_ch_in_group
                weight_offset = (group_idx * group_size * in_group_size * kernel_len + 
                                 out_ch_in_group * in_group_size * kernel_len + 
                                 in_ch_in_group * kernel_len + k)
                weight_val = tl.cast(tl.load(weight_ptr + weight_offset), tl.float32)
                in_val = tl.cast(tl.load(in_ptr + (batch * in_channels * in_len + 
                                           in_ch * in_len + in_base_pos)), tl.float32)
                acc = acc + weight_val * in_val
    
    # GELU activation using sigmoid approximation in fp32
    # gelu(x) ≈ x * sigmoid(1.702 * x)
    acc = acc * tl.sigmoid(tl.cast(1.702, tl.float32) * acc)
    
    # Cast back to input dtype for storage
    out_val = tl.cast(acc, tl.bfloat16)
    
    # Store result
    out_idx = batch * out_channels * out_len + out_ch * out_len + out_pos
    tl.store(out_ptr + out_idx, out_val)


@torch.fx.wrap
def fused_conv1d_gelu_slice(in_3, in_4, in_2):
    """Fused conv1d + gelu + slice[..., :124] kernel wrapper"""
    batch_size, in_channels, in_len = in_3.shape  # [1, 768, 249]
    out_channels, _, kernel_len = in_4.shape  # [768, 48, 31]
    
    # Compute output length: (in_len + 2*padding - dilation*(kernel_len-1) - 1) / stride + 1
    stride, padding, dilation = 2, 15, 1
    out_len_full = (in_len + 2 * padding - dilation * (kernel_len - 1) - 1) // stride + 1  # 125
    
    # Output shape after slicing [1, 48, 124]
    out_len = 124
    total_elements = batch_size * out_channels * out_len
    
    # Allocate output
    out = torch.empty((batch_size, out_channels, out_len), 
                      dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    BLOCK_SIZE = 128
    num_programs = total_elements
    
    fused_conv1d_gelu_slice_kernel[(num_programs,)](
        in_ptr=in_3, weight_ptr=in_4, bias_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels,
        in_len=in_len, kernel_len=kernel_len,
        stride=stride, padding=padding, dilation=dilation, groups=16,
        out_len=out_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# ============================================================================
# Fused AvgPool1d + Slice Pass
# ============================================================================
# Input shapes:
#   in_3: [1, 768, 249] - hidden_states
#
# Operations: avg_pool1d(in_3) -> slice[..., :124]
# Output: tmp_6 with shape [1, 768, 124]

@triton.jit
def fused_avgpool_slice_kernel(
    in_ptr, out_ptr,
    batch_size, channels, in_len,
    kernel_size, stride, padding,
    out_len,  # = 124
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // (channels * out_len)
    ch = (pid // out_len) % channels
    out_pos = pid % out_len
    
    # Input start position for average pooling
    in_start = out_pos * stride - padding
    
    # Accumulate for average pooling
    acc = 0.0
    count = 0
    for k in range(kernel_size):
        in_pos = in_start + k
        # Check bounds: need both conditions
        is_valid = (in_pos >= 0) and (in_pos < in_len)
        if is_valid:
            acc += tl.load(in_ptr + (batch * channels * in_len + ch * in_len + in_pos))
            count += 1
    
    # Average
    out_val = acc / tl.constexpr(kernel_size) if count == kernel_size else acc / count if count > 0 else 0.0
    
    # Store result
    out_idx = batch * channels * out_len + ch * out_len + out_pos
    tl.store(out_ptr + out_idx, out_val)


@torch.fx.wrap
def fused_avgpool_slice(in_3):
    """Fused avg_pool1d + slice[..., :124] kernel wrapper"""
    batch_size, channels, in_len = in_3.shape  # [1, 768, 249]
    
    # avg_pool1d parameters: kernel_size=2, stride=2, padding=0
    kernel_size, stride, padding = 2, 2, 0
    out_len_full = (in_len + 2 * padding - kernel_size) // stride + 1  # 124
    out_len = 124  # After slicing
    
    total_elements = batch_size * channels * out_len
    
    # Allocate output
    out = torch.empty((batch_size, channels, out_len), 
                      dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    BLOCK_SIZE = 128
    num_programs = total_elements
    
    fused_avgpool_slice_kernel[(num_programs,)](
        in_ptr=in_3, out_ptr=out,
        batch_size=batch_size, channels=channels, in_len=in_len,
        kernel_size=kernel_size, stride=stride, padding=padding,
        out_len=out_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# ============================================================================
# Fused Add + Transpose + LayerNorm Pass
# ============================================================================
# Input shapes:
#   tmp_6: [1, 768, 124]
#   tmp_7: [1, 48, 124]
#   in_0: [768] - layer_norm bias
#   in_1: [768] - layer_norm weight
#
# Operations: tmp_6 + tmp_7 -> transpose -> layer_norm
# Output: tmp_10 with shape [1, 124, 768]

@triton.jit
def fused_add_transpose_layernorm_kernel(
    a_ptr, b_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, dim_a, dim_b,  # dim_a=768, dim_b=48
    seq_len,  # 124
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Add: tmp_6 + tmp_7
    # tmp_6 is [batch, 768, 124], tmp_7 is [batch, 48, 124]
    # Broadcasting: tmp_7 [batch, 48, 124] broadcasts to tmp_6 [batch, 768, 124]
    # The addition happens on dim_b=48 (last dim), which broadcasts to dim_a=768
    
    # For each channel in dim_a (768):
    sum_vals = 0.0
    sq_sum = 0.0
    
    # First compute sum across the broadcast dimension (should be 1.0 weight per channel)
    # Actually, we need to handle the add carefully
    # tmp_6[..., :48] + tmp_7 -> tmp_8[..., :48], and tmp_6[..., 48:] stays the same
    
    # But since dim_a=768 and dim_b=48, broadcasting means tmp_7 only affects first 48 channels
    
    for ch in range(dim_a):
        # Load from tmp_6 [batch, ch, seq_idx]
        a_val = tl.load(a_ptr + (batch_idx * dim_a * seq_len + ch * seq_len + seq_idx))
        
        # Load from tmp_7 if in broadcast range [batch, ch, seq_idx]
        if ch < dim_b:
            b_val = tl.load(b_ptr + (batch_idx * dim_b * seq_len + ch * seq_len + seq_idx))
        else:
            b_val = 0.0
        
        # Add
        added_val = a_val + b_val
        
        # For layer norm: compute sum and sq_sum
        sum_vals += added_val
        sq_sum += added_val * added_val
    
    # Mean and variance
    mean = sum_vals / tl.cast(dim_a, tl.float32)
    variance = (sq_sum / tl.cast(dim_a, tl.float32)) - mean * mean
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize and affine transform
    for ch in range(dim_a):
        a_val = tl.load(a_ptr + (batch_idx * dim_a * seq_len + ch * seq_len + seq_idx))
        if ch < dim_b:
            b_val = tl.load(b_ptr + (batch_idx * dim_b * seq_len + ch * seq_len + seq_idx))
        else:
            b_val = 0.0
        
        added_val = a_val + b_val
        normalized = (added_val - mean) * inv_std
        
        # Affine transform: normalized * weight + bias
        w = tl.load(weight_ptr + ch)
        b = tl.load(bias_ptr + ch)
        out_val = normalized * w + b
        
        # Store output (transposed: [batch, seq, dim] -> [batch, seq, dim])
        out_idx = batch_idx * dim_a * seq_len + seq_idx * dim_a + ch
        tl.store(out_ptr + out_idx, out_val)


@torch.fx.wrap
def fused_add_transpose_layernorm(tmp_6, tmp_7, in_0, in_1, in_3):
    """Fused add + transpose + layer_norm kernel wrapper"""
    # tmp_6 shape: [1, 768, 124], tmp_7 shape: [1, 48, 124]
    batch, dim_a, seq_len = tmp_6.shape
    _, dim_b, _ = tmp_7.shape
    
    # Output shape (after transpose): [1, 124, 768]
    out = torch.empty((batch, seq_len, dim_a), 
                      dtype=tmp_6.dtype, device=tmp_6.device)
    
    total_elements = batch * seq_len
    
    # Launch kernel
    BLOCK_SIZE = 128
    num_programs = total_elements
    
    fused_add_transpose_layernorm_kernel[(num_programs,)](
        a_ptr=tmp_6, b_ptr=tmp_7, 
        weight_ptr=in_1, bias_ptr=in_0, 
        out_ptr=out,
        batch=batch, dim_a=dim_a, dim_b=dim_b,
        seq_len=seq_len,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# ============================================================================
# Pattern Matching Functions
# ============================================================================

def pattern(in_3, in_4, in_2):
    """Match: conv1d -> gelu -> slice[..., :124]"""
    conv1d = torch.conv1d(in_3, in_4, in_2, (2,), (15,), (1,), 16)
    tmp_4 = torch.nn.functional.gelu(conv1d)
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    return tmp_7


def replacement_args(in_3, in_4, in_2):
    """Extract arguments for the fused kernel"""
    return (in_3, in_4, in_2)


def replacement_func():
    """Return the fused conv1d+gelu+slice kernel"""
    return fused_conv1d_gelu_slice