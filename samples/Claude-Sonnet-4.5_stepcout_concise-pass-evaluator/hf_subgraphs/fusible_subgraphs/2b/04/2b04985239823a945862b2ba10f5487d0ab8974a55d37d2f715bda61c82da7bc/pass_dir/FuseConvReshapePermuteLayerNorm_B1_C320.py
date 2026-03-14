import torch
import triton
import triton.language as tl


def pattern(conv_out, ln_weight, ln_bias):
    """
    Pattern: reshape -> permute -> layer_norm
    Specific to batch=1, channels=320
    """
    reshaped = conv_out.reshape(1, 320, -1)
    permuted = reshaped.permute(0, 2, 1)
    normalized = torch.nn.functional.layer_norm(permuted, (320,), ln_weight, ln_bias, 1e-05)
    
    return normalized


def replacement_args(conv_out, ln_weight, ln_bias):
    return (conv_out, ln_weight, ln_bias)


@triton.jit
def fused_kernel_b1_c320(
    conv_out_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    batch_size,
    seq_len,
    channels,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for reshape + permute + layer_norm.
    """
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    if seq_idx >= seq_len or batch_idx >= batch_size:
        return
    
    # Calculate mean
    mean = 0.0
    for c_start in range(0, channels, BLOCK_SIZE):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE)
        c_mask = c_offsets < channels
        
        conv_out_offsets = batch_idx * channels * seq_len + c_offsets * seq_len + seq_idx
        vals = tl.load(conv_out_ptr + conv_out_offsets, mask=c_mask, other=0.0)
        mean += tl.sum(vals)
    
    mean = mean / channels
    
    # Calculate variance
    var = 0.0
    for c_start in range(0, channels, BLOCK_SIZE):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE)
        c_mask = c_offsets < channels
        
        conv_out_offsets = batch_idx * channels * seq_len + c_offsets * seq_len + seq_idx
        vals = tl.load(conv_out_ptr + conv_out_offsets, mask=c_mask, other=0.0)
        diff = vals - mean
        var += tl.sum(diff * diff)
    
    var = var / channels
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine transform
    for c_start in range(0, channels, BLOCK_SIZE):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE)
        c_mask = c_offsets < channels
        
        conv_out_offsets = batch_idx * channels * seq_len + c_offsets * seq_len + seq_idx
        vals = tl.load(conv_out_ptr + conv_out_offsets, mask=c_mask, other=0.0)
        
        weight = tl.load(ln_weight_ptr + c_offsets, mask=c_mask, other=1.0)
        bias = tl.load(ln_bias_ptr + c_offsets, mask=c_mask, other=0.0)
        
        normalized = (vals - mean) * rstd * weight + bias
        
        output_offsets = batch_idx * seq_len * channels + seq_idx * channels + c_offsets
        tl.store(output_ptr + output_offsets, normalized, mask=c_mask)


@torch.fx.wrap
def wrapper_b1_c320(conv_out, ln_weight, ln_bias):
    """
    Wrapper for batch=1, channels=320
    """
    batch_size, channels, H, W = conv_out.shape
    seq_len = H * W
    
    conv_out_flat = conv_out.reshape(batch_size, channels, seq_len).contiguous()
    output = torch.empty(batch_size, seq_len, channels, dtype=conv_out.dtype, device=conv_out.device)
    
    BLOCK_SIZE = 128
    grid = (seq_len, batch_size)
    
    fused_kernel_b1_c320[grid](
        conv_out_flat,
        ln_weight,
        ln_bias,
        output,
        batch_size,
        seq_len,
        channels,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return wrapper_b1_c320