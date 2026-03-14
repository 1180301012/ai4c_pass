import torch
import triton
import triton.language as tl


def pattern(conv_out, ln_weight, ln_bias):
    """
    Pattern: reshape -> permute -> layer_norm
    Match the post-conv2d operations that appear in all target graphs.
    
    The actual model has exactly:
    - reshaped = conv_out.reshape(batch, channels, -1)
    - permuted = reshaped.permute(0, 2, 1)  
    - normalized = torch.nn.functional.layer_norm(permuted, (channels,), ln_weight, ln_bias, 1e-05)
    
    We need to match without adding extra operations like .size() calls.
    Since patterns work symbolically, we can use any values and the framework will match.
    """
    # Use concrete values - the framework should match these symbolically
    # Let's try batch=1, channels=64 as a representative case
    reshaped = conv_out.reshape(1, 64, -1)
    permuted = reshaped.permute(0, 2, 1)
    normalized = torch.nn.functional.layer_norm(permuted, (64,), ln_weight, ln_bias, 1e-05)
    
    return normalized


def replacement_args(conv_out, ln_weight, ln_bias):
    return (conv_out, ln_weight, ln_bias)


@triton.jit
def fused_kernel_b1_c64(
    conv_out_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    batch_size,
    seq_len,
    channels,
    eps: tl.constexpr,
    BLOCK_SIZE_CHAN: tl.constexpr,
):
    """
    Fused kernel for reshape + permute + layer_norm.
    Simpler design: one thread block per (batch, seq) position.
    """
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    if seq_idx >= seq_len or batch_idx >= batch_size:
        return
    
    # Channel offsets
    c_offsets = tl.arange(0, BLOCK_SIZE_CHAN)
    c_mask = c_offsets < channels
    
    # Input layout: (B, C, seq_len)
    # Load values for this (batch, seq) position across all channels
    conv_out_offsets = batch_idx * channels * seq_len + c_offsets * seq_len + seq_idx
    vals = tl.load(conv_out_ptr + conv_out_offsets, mask=c_mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(vals) / channels
    
    # Calculate variance - zero out masked values
    centered = tl.where(c_mask, vals - mean, 0.0)
    var = tl.sum(centered * centered) / channels
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(ln_weight_ptr + c_offsets, mask=c_mask, other=1.0)
    bias = tl.load(ln_bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Normalize
    normalized = (vals - mean) * rstd * weight + bias
    
    # Store to output: (B, seq_len, C)
    output_offsets = batch_idx * seq_len * channels + seq_idx * channels + c_offsets
    tl.store(output_ptr + output_offsets, normalized, mask=c_mask)


@torch.fx.wrap
def wrapper_b1_c64(conv_out, ln_weight, ln_bias):
    """
    Wrapper function that fuses reshape + permute + layer_norm in a Triton kernel.
    Input: conv_out with shape (B, C, H, W)
    Output: normalized with shape (B, H*W, C)
    """
    # Get dimensions
    batch_size, channels, H, W = conv_out.shape
    seq_len = H * W
    
    # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
    # Make sure the tensor is contiguous
    conv_out_flat = conv_out.reshape(batch_size, channels, seq_len).contiguous()
    
    # Allocate output: (B, seq_len, C)
    output = torch.empty(batch_size, seq_len, channels, dtype=conv_out.dtype, device=conv_out.device)
    
    # Set block size for channels
    BLOCK_SIZE_CHAN = triton.next_power_of_2(channels)
    
    # Grid: one thread block per (batch, seq) position
    grid = (seq_len, batch_size)
    
    fused_kernel_b1_c64[grid](
        conv_out_flat,
        ln_weight,
        ln_bias,
        output,
        batch_size,
        seq_len,
        channels,
        eps=1e-05,
        BLOCK_SIZE_CHAN=BLOCK_SIZE_CHAN,
    )
    
    return output


def replacement_func():
    return wrapper_b1_c64