import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match: Add + LayerNorm + Slice + Reshape + Permute + ReLU
    """
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1280,), in_1, in_0, 1e-06)
    tmp_4 = tmp_3[slice(None, None, None), slice(0, None, None)]
    tmp_5 = tmp_4.reshape(tmp_4.shape[0], 16, 12, -1)
    tmp_6 = tmp_5.permute(0, 3, 1, 2)
    tmp_7 = torch.nn.functional.relu(tmp_6)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_ln_permute_relu_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, seq_len, hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for Add + LayerNorm + Reshape + Permute + ReLU
    
    Input shape: [batch, seq_len, hidden_dim] = [batch, 192, 1280]
    Output shape: [batch, hidden_dim, 16, 12] = [batch, 1280, 16, 12]
    
    Each program handles one (batch, seq) position and processes all channels.
    The output is written directly in the permuted layout.
    """
    pid = tl.program_id(0)
    
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Compute output h, w from seq_idx (reshape maps [batch, 192, 1280] -> [batch, 16, 12, 1280])
    # seq_idx = h * 12 + w
    h = seq_idx // 12
    w = seq_idx % 12
    
    # Base offset for this (batch, seq) position
    offset_base = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
    
    # First pass: compute mean and variance over all channels
    sum_x = 0.0
    sum_x2 = 0.0
    
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Load inputs
        in_2 = tl.load(in_2_ptr + offset_base + offsets, mask=mask, other=0.0)
        in_3 = tl.load(in_3_ptr + offset_base + offsets, mask=mask, other=0.0)
        
        # Add
        x = in_2 + in_3
        
        # Accumulate statistics
        sum_x += tl.sum(tl.where(mask, x, 0.0))
        sum_x2 += tl.sum(tl.where(mask, x * x, 0.0))
    
    # Compute mean and std
    mean = sum_x / hidden_dim
    var = sum_x2 / hidden_dim - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: normalize, apply ReLU, write to output in permuted layout
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        
        # Load inputs again
        in_2 = tl.load(in_2_ptr + offset_base + offsets, mask=mask, other=0.0)
        in_3 = tl.load(in_3_ptr + offset_base + offsets, mask=mask, other=0.0)
        x = in_2 + in_3
        
        # Load weight and bias for layer norm
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        # Apply layer normalization
        normalized = (x - mean) * rstd * w + b
        
        # Apply ReLU
        result = tl.maximum(normalized, 0.0)
        
        # Write to output in permuted layout
        # Output shape: [batch, hidden_dim, 16, 12]
        # Output[batch_idx, channel, h, w] = result[channel]
        # Linear index: batch_idx * (hidden_dim * 16 * 12) + channel * (16 * 12) + h * 12 + w
        channel_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        out_offsets = (batch_idx * (hidden_dim * 16 * 12) + 
                       channel_offsets * (16 * 12) + 
                       h * 12 + w)
        tl.store(out_ptr + out_offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_ln_permute_relu(in_0, in_1, in_2, in_3):
    """
    Wrapper function for the fused kernel
    in_0: bias
    in_1: weight
    in_2: input_160
    in_3: x_163
    """
    bias = in_0
    weight = in_1
    
    batch, seq_len, hidden_dim = in_2.shape
    
    # Output shape after reshape and permute: [batch, hidden_dim, 16, 12]
    output = torch.empty((batch, hidden_dim, 16, 12), dtype=in_2.dtype, device=in_2.device)
    
    # Grid: one program per (batch, seq) position
    grid = (batch * seq_len,)
    
    # Launch kernel
    BLOCK_SIZE = 256
    eps = 1e-06
    
    fused_add_ln_permute_relu_kernel[grid](
        in_2, in_3, weight, bias, output,
        batch, seq_len, hidden_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_add_ln_permute_relu