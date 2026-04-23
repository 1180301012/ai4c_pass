import torch
import triton
import triton.language as tl


@triton.jit
def fused_mul_bn_silu_kernel(
    x_ptr,
    sigmoid_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    n_channels,
    n_hw,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of elements
    batch_pid = tl.program_id(0)
    channel_pid = tl.program_id(1)
    
    # Calculate offsets
    n_elements_per_batch = n_channels * n_hw
    batch_start = batch_pid * n_elements_per_batch
    channel_offset = channel_pid * n_hw
    
    # Offsets for this program
    offs = batch_start + channel_offset + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load x: [B, C, H, W] at position (batch, channel, h, w) within block
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Load sigmoid: [B, C, 1, 1] - need to broadcast to channel
    # sigmoid value is same for all h,w within a channel
    sigmoid_offs = batch_pid * n_channels + channel_pid
    sigmoid_val = tl.load(sigmoid_ptr + sigmoid_offs)
    
    # Step 1: Multiply x by sigmoid (broadcast)
    x_mul = x * sigmoid_val
    
    # Load batch norm parameters (per channel)
    mean = tl.load(running_mean_ptr + channel_pid)
    var = tl.load(running_var_ptr + channel_pid)
    weight = tl.load(weight_ptr + channel_pid)
    bias = tl.load(bias_ptr + channel_pid)
    
    # Step 2: Batch normalization
    # std = sqrt(var + eps)
    inv_std = 1.0 / tl.sqrt(var + eps)
    # normalized = (x - mean) * inv_std
    x_norm = (x_mul - mean) * inv_std
    # output = normalized * weight + bias
    bn_out = x_norm * weight + bias
    
    # Step 3: SiLU activation (inplace on result)
    # silu(x) = x * sigmoid(x)
    silu_out = bn_out * tl.sigmoid(bn_out)
    
    # Store result
    tl.store(output_ptr + offs, silu_out, mask=mask)


@torch.fx.wrap
def fused_mul_bn_silu(
    x,
    sigmoid,
    running_mean,
    running_var,
    weight,
    bias,
    eps=1e-05,
    momentum=0.1
):
    """
    Fused kernel: (x * sigmoid) -> batch_norm -> silu
    """
    B, C, H, W = x.shape
    n_elements = B * C * H * W
    n_hw = H * W
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Grid configuration
    BLOCK_SIZE = 1024
    # Use 2D grid: (batch, channel)
    grid = (B, C)
    
    fused_mul_bn_silu_kernel[grid](
        x,
        sigmoid,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        n_elements,
        C,
        n_hw,
        eps,
        momentum,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: (in_5 * in_4) -> batch_norm -> silu
    Returns the intermediate values for proper subgraph matching.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_6 = tmp_5 * torch.sigmoid(tmp_5)
    return tmp_4, tmp_5, tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract arguments needed for the fused operation.
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_mul_bn_silu