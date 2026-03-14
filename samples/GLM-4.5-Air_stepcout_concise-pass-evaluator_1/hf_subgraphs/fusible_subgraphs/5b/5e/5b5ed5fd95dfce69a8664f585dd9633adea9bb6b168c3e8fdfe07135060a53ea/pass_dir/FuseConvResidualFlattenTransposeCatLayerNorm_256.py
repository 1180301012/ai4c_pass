import torch
import triton
import triton.language as tl


# Pattern for Conv2D with groups=256 + residual + flatten + transpose + cat + layer_norm
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match: Conv2D (groups=256) + Add + Flatten + Transpose + Cat + LayerNorm"""
    tmp_4 = torch.conv2d(in_5, in_1, in_0, (1, 1), (1, 1), (1, 1), 256)
    tmp_5 = tmp_4 + in_5
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.cat((in_4, tmp_7), dim=1)
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (256,), in_3, in_2, 1e-06)
    return tmp_8, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1}, num_stages=3, num_warps=1),
        triton.Config({'BLOCK_C': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C': 8}, num_stages=3, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def fused_cpe_layernorm_kernel_256(
    input_ptr, weight_ptr, bias_ptr,
    ln_weight_ptr, ln_bias_ptr, cls_token_ptr,
    output_ptr, ln_output_ptr,
    B, C, H, W, num_tokens,
    stride_input, stride_weight, stride_bias,
    stride_ln_w, stride_ln_b, stride_cls,
    stride_out, stride_ln_out,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_channel_blocks = tl.cdiv(C, BLOCK_C)
    token_pid = pid // num_channel_blocks
    channel_block = pid % num_channel_blocks
    
    if token_pid >= B * num_tokens:
        return
    
    b = token_pid // num_tokens
    token = token_pid % num_tokens
    
    if token == 0:
        for cb in range(BLOCK_C):
            c = channel_block * BLOCK_C + cb
            if c >= C:
                break
            cls_val = tl.load(cls_token_ptr + c)
            out_idx = b * stride_out + 0 * C + c
            tl.store(output_ptr + out_idx, cls_val)
            ln_out_idx = b * stride_ln_out + 0 * C + c
            tl.store(ln_output_ptr + ln_out_idx, cls_val)
    else:
        spatial_idx = token - 1
        spatial_h = spatial_idx // W
        spatial_w = spatial_idx % W
        
        for cb in range(BLOCK_C):
            c = channel_block * BLOCK_C + cb
            if c >= C:
                break
            
            conv_out = 0.0
            weight_base = c * 9
            
            for kh in range(3):
                for kw in range(3):
                    ih = spatial_h + kh - 1
                    iw = spatial_w + kw - 1
                    if 0 <= ih < H and 0 <= iw < W:
                        wk = kh * 3 + kw
                        weight_val = tl.load(weight_ptr + weight_base + wk)
                        input_idx = b * stride_input + c * H * W + ih * W + iw
                        input_val = tl.load(input_ptr + input_idx)
                        conv_out += input_val * weight_val
            
            bias_val = tl.load(bias_ptr + c)
            conv_out += bias_val
            
            residual_idx = b * stride_input + c * H * W + spatial_h * W + spatial_w
            residual_val = tl.load(input_ptr + residual_idx)
            out_val = conv_out + residual_val
            
            out_idx = b * stride_out + token * C + c
            tl.store(output_ptr + out_idx, out_val)
            
            ln_out_idx = b * stride_ln_out + token * C + c
            tl.store(ln_output_ptr + ln_out_idx, out_val)


@torch.fx.wrap
def fused_kernel_wrapper_256(in_0, in_1, in_2, in_3, in_4, in_5):
    B, C, H, W = in_5.shape
    num_tokens = H * W + 1
    out_channels = in_1.shape[0]
    
    output = torch.empty(B, num_tokens, C, device=in_5.device, dtype=in_5.dtype)
    ln_output = torch.empty(B, num_tokens, C, device=in_5.device, dtype=in_5.dtype)
    
    num_channel_blocks = triton.cdiv(C, 8)
    grid = (B * num_tokens * num_channel_blocks,)
    
    fused_cpe_layernorm_kernel_256[grid](
        in_5, in_1, in_0,
        in_3, in_2, in_4.squeeze(0).squeeze(0),
        output, ln_output,
        B, C, H, W, num_tokens,
        in_5.stride(0), in_1.stride(0), in_0.stride(0),
        in_3.stride(0), in_2.stride(0), in_4.stride(0),
        output.stride(0), ln_output.stride(0),
        1e-6,
        8,
    )
    
    return output, ln_output


def replacement_func():
    return fused_kernel_wrapper_256