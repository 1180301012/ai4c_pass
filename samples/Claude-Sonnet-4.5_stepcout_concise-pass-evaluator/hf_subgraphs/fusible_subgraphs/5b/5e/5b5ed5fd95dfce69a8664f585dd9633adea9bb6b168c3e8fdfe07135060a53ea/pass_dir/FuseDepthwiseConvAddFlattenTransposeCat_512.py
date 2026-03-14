import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern matching for depthwise conv + add + flatten + transpose + cat (512 channels)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.conv2d(in_5, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 512)
    tmp_5 = tmp_4 + in_5
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.cat((in_4, tmp_7), dim=1)
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (512,), tmp_3, tmp_2, 1e-06)
    return (tmp_8, tmp_9)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def depthwise_conv_add_kernel_512(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C, H, W,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_wc, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output position
    pid = tl.program_id(0)
    
    # Compute which batch, channel, h, w this program handles
    num_elements = B * C * H * W
    if pid >= num_elements:
        return
    
    b = pid // (C * H * W)
    remainder = pid % (C * H * W)
    c = remainder // (H * W)
    remainder = remainder % (H * W)
    h = remainder // W
    w = remainder % W
    
    # Depthwise conv 3x3 with padding=1
    result = 0.0
    for kh in range(3):
        for kw in range(3):
            ih = h + kh - 1
            iw = w + kw - 1
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                input_offset = b * stride_ib + c * stride_ic + ih * stride_ih + iw * stride_iw
                weight_offset = c * stride_wc + kh * stride_wh + kw * stride_ww
                input_val = tl.load(input_ptr + input_offset)
                weight_val = tl.load(weight_ptr + weight_offset)
                result += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + c)
    result += bias_val
    
    # Add residual (input)
    input_offset = b * stride_ib + c * stride_ic + h * stride_ih + w * stride_iw
    input_val = tl.load(input_ptr + input_offset)
    result += input_val
    
    # Store output
    output_offset = b * stride_ob + c * stride_oc + h * stride_oh + w * stride_ow
    tl.store(output_ptr + output_offset, result)

@triton.jit
def flatten_transpose_cat_kernel_512(
    conv_output_ptr, cls_token_ptr, output_ptr,
    B, C, H, W,
    stride_cb, stride_cc, stride_ch, stride_cw,
    stride_clsb, stride_clss, stride_clsc,
    stride_ob, stride_os, stride_oc,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Output shape: [B, H*W+1, C]
    seq_len = H * W + 1
    num_elements = B * seq_len * C
    
    if pid >= num_elements:
        return
    
    b = pid // (seq_len * C)
    remainder = pid % (seq_len * C)
    s = remainder // C
    c = remainder % C
    
    if s == 0:
        # Load from cls_token
        cls_offset = b * stride_clsb + 0 * stride_clss + c * stride_clsc
        val = tl.load(cls_token_ptr + cls_offset)
    else:
        # Load from conv_output (flattened and transposed)
        spatial_idx = s - 1
        h = spatial_idx // W
        w = spatial_idx % W
        conv_offset = b * stride_cb + c * stride_cc + h * stride_ch + w * stride_cw
        val = tl.load(conv_output_ptr + conv_offset)
    
    # Store to output
    output_offset = b * stride_ob + s * stride_os + c * stride_oc
    tl.store(output_ptr + output_offset, val)

@triton.jit
def layer_norm_kernel_512(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, S, C,
    stride_ib, stride_is, stride_ic,
    stride_ob, stride_os, stride_oc,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sequence position
    pid = tl.program_id(0)
    
    if pid >= B * S:
        return
    
    b = pid // S
    s = pid % S
    
    # Compute mean
    mean = 0.0
    for c in range(C):
        offset = b * stride_ib + s * stride_is + c * stride_ic
        val = tl.load(input_ptr + offset)
        mean += val
    mean = mean / C
    
    # Compute variance
    var = 0.0
    for c in range(C):
        offset = b * stride_ib + s * stride_is + c * stride_ic
        val = tl.load(input_ptr + offset)
        diff = val - mean
        var += diff * diff
    var = var / C
    
    # Normalize and apply weight/bias
    for c in range(C):
        input_offset = b * stride_ib + s * stride_is + c * stride_ic
        val = tl.load(input_ptr + input_offset)
        
        normalized = (val - mean) / tl.sqrt(var + eps)
        
        weight_val = tl.load(weight_ptr + c)
        bias_val = tl.load(bias_ptr + c)
        
        output_val = normalized * weight_val + bias_val
        
        output_offset = b * stride_ob + s * stride_os + c * stride_oc
        tl.store(output_ptr + output_offset, output_val)

@torch.fx.wrap
def fused_depthwise_conv_add_flatten_transpose_cat_layernorm_512(in_0, in_1, in_2, in_3, in_4, in_5):
    B, C, H, W = in_5.shape
    
    # Step 1: Depthwise conv + add
    conv_output = torch.empty_like(in_5)
    num_elements = B * C * H * W
    grid = (num_elements,)
    depthwise_conv_add_kernel_512[grid](
        in_5, in_1, in_0, conv_output,
        B, C, H, W,
        in_5.stride(0), in_5.stride(1), in_5.stride(2), in_5.stride(3),
        in_1.stride(0), in_1.stride(2), in_1.stride(3),
        conv_output.stride(0), conv_output.stride(1), conv_output.stride(2), conv_output.stride(3),
        BLOCK_SIZE=1024,
    )
    
    # Step 2: Flatten + transpose + cat
    seq_len = H * W + 1
    tmp_8 = torch.empty(B, seq_len, C, device=in_5.device, dtype=in_5.dtype)
    num_elements = B * seq_len * C
    grid = (num_elements,)
    flatten_transpose_cat_kernel_512[grid](
        conv_output, in_4, tmp_8,
        B, C, H, W,
        conv_output.stride(0), conv_output.stride(1), conv_output.stride(2), conv_output.stride(3),
        in_4.stride(0), in_4.stride(1), in_4.stride(2),
        tmp_8.stride(0), tmp_8.stride(1), tmp_8.stride(2),
        BLOCK_SIZE=1024,
    )
    
    # Step 3: Layer norm
    tmp_9 = torch.empty_like(tmp_8)
    grid = (B * seq_len,)
    layer_norm_kernel_512[grid](
        tmp_8, in_3, in_2, tmp_9,
        B, seq_len, C,
        tmp_8.stride(0), tmp_8.stride(1), tmp_8.stride(2),
        tmp_9.stride(0), tmp_9.stride(1), tmp_9.stride(2),
        eps=1e-06,
        BLOCK_SIZE=1024,
    )
    
    return (tmp_8, tmp_9)

def replacement_func():
    return fused_depthwise_conv_add_flatten_transpose_cat_layernorm_512