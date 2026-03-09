import torch
import triton
import triton.language as tl

# Pattern: flatten(2) + transpose(1,2) + layer_norm + view + permute(0,3,1,2)
# This fuses: input[B, C, H, W] -> flatten -> transpose -> layernorm -> view -> permute -> output[C, B, H, W]
def pattern(in_5, in_3, in_4):
    # in_5 is the fused conv output: [B, C, H, W]
    # flatten(2): [B, C, H, W] -> [B, C, H*W]
    tmp_11 = in_5.flatten(2)
    # transpose: [B, C, H*W] -> [B, H*W, C]
    tmp_12 = tmp_11.transpose(1, 2)
    # layer_norm on normalized_shape=C
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (in_4.shape[0],), in_4, in_3, 1e-06)
    # view: [B, H*W, C] -> [B, H, W, C]
    # The view shape varies based on input, e.g., [256, 56, 56, 16] or [1, 56, 56, 64] or [64, 7, 7, 128]
    # But we can get the original H, W from in_5
    B, C, H, W = in_5.shape
    tmp_14 = tmp_13.view(B, H, W, C)
    # permute: [B, H, W, C] -> [C, B, H, W]
    tmp_15 = tmp_14.permute(0, 3, 1, 2)
    return tmp_15

def replacement_args(in_5, in_3, in_4):
    return (in_5, in_3, in_4)


def pattern2(in_6, in_3, in_4):
    # Same pattern but with in_6 as input (for the variant where conv uses in_6)
    tmp_11 = in_6.flatten(2)
    tmp_12 = tmp_11.transpose(1, 2)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (in_4.shape[0],), in_4, in_3, 1e-06)
    B, C, H, W = in_6.shape
    tmp_14 = tmp_13.view(B, H, W, C)
    tmp_15 = tmp_14.permute(0, 3, 1, 2)
    return tmp_15

def replacement_args2(in_6, in_3, in_4):
    return (in_6, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_layernorm_reshape_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C, H, W,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_w, stride_out_c, stride_out_b, stride_out_h, stride_out_w,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Grid: (B, H*W) - each block computes layernorm for one spatial position across all channels
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Compute the linearized spatial position
    h = spatial_idx // W
    w = spatial_idx % W
    
    # Pointers to the start of each spatial position's sequence: [B, H*W, C]
    # Input shape is [B, C, H, W]
    # After flatten(2), it becomes [B, C, H*W]
    # After transpose, it's [B, H*W, C]
    # So for spatial position (h,w), the sequence starts at offset batch_idx * B * C + spatial_idx * C
    
    # But we need to read from original [B, C, H, W] format
    # The layernorm computes stats across channel dimension C
    
    # Instead of materializing transpose, we read directly: for each channel c
    # input[b, c, h, w] -> we need sequence of [c=0..C-1] for given b, h, w
    
    # Load the channel values for this (b, h, w)
    # input[b, c, h, w] at offset: b*stride_in_b + c*stride_in_c + h*stride_in_h + w*stride_in_w
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < C
    
    # Load values: [C]
    base_offset = batch_idx * stride_in_b + h * stride_in_h + w * stride_in_w
    vals = tl.load(input_ptr + base_offset + offsets * stride_in_c, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(vals, axis=0) / C
    # Compute variance: var = E[(x - mean)^2]
    diff = vals - mean
    variance = tl.sum(diff * diff, axis=0) / C
    # std = sqrt(var + eps)
    eps = 1e-06
    std = tl.sqrt(variance + eps)
    
    # Normalize: (x - mean) / std * weight + bias
    normalized = diff / std
    
    # Load weight and bias [C]
    weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias_vals = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    output_vals = normalized * weight_vals + bias_vals
    
    # Now we need to permute and store: [C] -> [C, B, H, W] with permute(0,3,1,2)
    # So for output, we store at [c, b, h, w]
    # Output stride: [stride_out_c, stride_out_b, stride_out_h, stride_out_w]
    # offset = c * stride_out_c + b * stride_out_b + h * stride_out_h + w * stride_out_w
    
    # Store each channel to its position
    out_base = h * stride_out_h + w * stride_out_w
    
    for c in range(0, C):
        out_offset = c * stride_out_c + out_base
        tl.store(output_ptr + out_offset, output_vals[c])


@torch.fx.wrap
def fused_layernorm_reshape_wrapper(in_5, in_3, in_4):
    # in_5: fused conv output [B, C, H, W]
    # in_3: layernorm bias [C]
    # in_4: layernorm weight [C]
    
    B, C, H, W = in_5.shape
    
    # Output: [C, B, H, W]
    output = torch.empty((C, B, H, W), device=in_5.device, dtype=in_5.dtype)
    
    # Strides
    stride_in_b, stride_in_c, stride_in_h, stride_in_w = in_5.stride()
    stride_out_c, stride_out_b, stride_out_h, stride_out_w = output.stride()
    
    N = B * H * W
    
    # Grid
    grid = (B, H * W)
    
    fused_layernorm_reshape_kernel[grid](
        in_5, in_4, in_3, output,
        B, C, H, W,
        stride_in_b, stride_in_c, stride_in_h, stride_in_w,
        0, stride_out_c, stride_out_b, stride_out_h, stride_out_w,
        N,
    )
    
    return output


def replacement_func():
    return fused_layernorm_reshape_wrapper