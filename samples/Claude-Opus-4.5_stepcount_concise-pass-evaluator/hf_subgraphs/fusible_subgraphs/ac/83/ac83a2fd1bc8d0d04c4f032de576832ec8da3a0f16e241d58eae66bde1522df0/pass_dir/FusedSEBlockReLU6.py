import torch
import triton
import triton.language as tl

# Pattern matching function - must match model.py exactly
def pattern(bias, weight, x, x_se):
    conv_out = torch.conv2d(x_se, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    sigmoid_out = conv_out.sigmoid()
    mul_out = x * sigmoid_out
    result = torch.nn.functional.hardtanh(mul_out, 0.0, 6.0, False)
    return result

def replacement_args(bias, weight, x, x_se):
    return (bias, weight, x, x_se)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C_IN': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C_IN': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C_IN': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C_IN': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C_IN': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C_IN': 32}, num_warps=2, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit
def fused_se_relu6_kernel(
    x_ptr,
    x_se_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B, C_out, HW, C_in,
    stride_bc,
    BLOCK_HW: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Grid: (B * C_out, num_hw_blocks)
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    batch_idx = pid_bc // C_out
    channel_idx = pid_bc % C_out
    
    # Load bias
    acc = tl.load(bias_ptr + channel_idx).to(tl.float32)
    
    # Vectorized dot product
    c_in_offsets = tl.arange(0, BLOCK_C_IN)
    c_in_mask = c_in_offsets < C_in
    
    x_se_vals = tl.load(x_se_ptr + batch_idx * C_in + c_in_offsets, mask=c_in_mask, other=0.0)
    w_vals = tl.load(weight_ptr + channel_idx * C_in + c_in_offsets, mask=c_in_mask, other=0.0)
    acc += tl.sum(x_se_vals * w_vals, axis=0)
    
    # Sigmoid
    sigmoid_out = tl.sigmoid(acc)
    
    # Process spatial block - use flat indexing for contiguous tensors
    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW
    
    # Flat offset for contiguous NCHW layout
    base_offset = batch_idx * stride_bc + channel_idx * HW
    
    # Load, compute, store
    x_vals = tl.load(x_ptr + base_offset + hw_offsets, mask=hw_mask)
    result = x_vals * sigmoid_out
    result = tl.minimum(tl.maximum(result, 0.0), 6.0)
    tl.store(out_ptr + base_offset + hw_offsets, result, mask=hw_mask)


@torch.fx.wrap
def fused_se_relu6(bias, weight, x, x_se):
    B, C_out, H, W = x.shape
    C_in = x_se.shape[1]
    HW = H * W
    
    weight = weight.to(x.device)
    bias = bias.to(x.device)
    
    x_se_flat = x_se.view(B, C_in)
    weight_flat = weight.view(C_out, C_in)
    
    # Make contiguous and flatten spatial dims
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    
    def grid(meta):
        return (B * C_out, triton.cdiv(HW, meta['BLOCK_HW']))
    
    fused_se_relu6_kernel[grid](
        x_contig,
        x_se_flat,
        weight_flat,
        bias,
        out,
        B, C_out, HW, C_in,
        C_out * HW,
    )
    
    return out


def replacement_func():
    return fused_se_relu6