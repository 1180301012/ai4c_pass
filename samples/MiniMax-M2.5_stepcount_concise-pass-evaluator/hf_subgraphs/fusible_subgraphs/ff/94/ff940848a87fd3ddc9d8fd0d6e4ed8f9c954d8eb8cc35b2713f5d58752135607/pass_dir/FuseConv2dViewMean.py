import torch
import triton
import triton.language as tl


# Autotune configurations for different batch sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 16, 'num_warps': 1}, num_stages=1),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 32, 'num_warps': 2}, num_stages=1),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 64, 'num_warps': 4}, num_stages=1),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 32, 'num_warps': 2}, num_stages=1),
    ],
    key=['batch_size', 'out_channels'],
)
@triton.jit
def fused_conv_view_mean_kernel(
    # Conv parameters
    input_ptr,      # [batch, 512, 64, 64]
    weight_ptr,     # [256, 512, 1, 1]
    bias_ptr,       # [256]
    # Mean input
    mean_input_ptr, # [batch, 4096, 256]
    # Outputs
    conv_out_ptr,   # [batch, 256, 4096]
    mean_out_ptr,   # [batch, 1, 256]
    # Shapes
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel for:
    1. Conv2d: [batch, 512, 64, 64] x [256, 512, 1, 1] + [256] -> [batch, 256, 64, 64]
    2. View: [batch, 256, 64, 64] -> [batch, 256, 4096]
    3. Mean: [batch, 4096, 256] -> [batch, 1, 256]
    
    This kernel computes conv+view+mean in a single pass to avoid intermediate memory.
    """
    # Get program IDs
    pid_b = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1)  # output channel index
    
    # Conv is 1x1 with stride=1, padding=0, dilation=1
    # This is equivalent to: output[b, oc, h, w] = sum(ic) input[b, ic, h, w] * weight[oc, ic, 0, 0] + bias[oc]
    
    # For conv output at [b, oc, h, w], we need to compute:
    # sum over ic: input[b, ic, h, w] * weight[oc, ic]
    
    # Each program handles one (batch, out_channel) pair, iterating over H*W
    # Since H=W=64, H*W=4096
    
    # Load bias
    bias = tl.load(bias_ptr + pid_oc)
    
    # Initialize accumulator for conv
    conv_sum = 0.0
    
    # Iterate over input channels
    for ic in range(0, in_channels, BLOCK_N):
        ic_offsets = ic + tl.arange(0, BLOCK_N)
        ic_mask = ic_offsets < in_channels
        
        # Load weight: [out_channel, in_channels] -> we need row pid_oc
        weight_offsets = pid_oc * in_channels + ic_offsets
        weight = tl.load(weight_ptr + weight_offsets, mask=ic_mask, other=0.0)
        
        # Load input: [batch, in_channels, height, width]
        # We need input[b, ic_offsets, h, w] for all h, w
        # For each (h, w), we compute sum over ic
        
        # Iterate over height
        for h in range(0, height):
            for w in range(0, width):
                # Input offset: b * (in_channels * height * width) + ic * (height * width) + h * width + w
                base_offset = pid_b * in_channels * height * width + h * width + w
                
                # Load input for all ic
                input_offsets = base_offset + ic_offsets * height * width
                input_vals = tl.load(input_ptr + input_offsets, mask=ic_mask, other=0.0)
                
                # Accumulate: sum(input * weight)
                conv_sum += tl.sum(input_vals * weight, axis=0)
    
    # Add bias
    conv_result = conv_sum + bias
    
    # Store conv output (after view: [batch, 256, 64, 64] -> [batch, 256, 4096])
    # The output is stored at [batch, oc, h*w + w]
    for h in range(0, height):
        for w in range(0, width):
            view_idx = h * width + w
            out_offset = pid_b * out_channels * height * width + pid_oc * height * width + view_idx
            tl.store(conv_out_ptr + out_offset, conv_result)
    
    # Now compute mean over dim=-2 (the 4096 dimension)
    # mean_input is [batch, 4096, 256]
    # We need mean over dim=-2, which is dimension 1 (size 4096)
    
    # For each batch b and channel k, compute mean over the 4096 elements
    # This program handles (b, k) = (pid_b, pid_oc)
    
    mean_sum = 0.0
    
    # mean_input is stored as [b, 4096, 256] = [b, h*w, k]
    # We need mean over the 4096 elements for channel pid_oc
    mean_base = pid_b * 4096 * 256 + pid_oc
    
    # Iterate over the 4096 elements
    # Each iteration loads BLOCK_M elements
    for idx in range(0, 4096, BLOCK_M):
        idx_offsets = idx + tl.arange(0, BLOCK_M)
        mask = idx_offsets < 4096
        
        # Load: mean_input[b, idx_offsets, k]
        # Offset = b * 4096 * 256 + idx_offsets * 256 + k
        mean_offsets = mean_base + idx_offsets * 256
        mean_vals = tl.load(mean_input_ptr + mean_offsets, mask=mask, other=0.0)
        
        mean_sum += tl.sum(mean_vals, axis=0)
    
    mean_result = mean_sum / 4096
    
    # Store mean output: [batch, 1, 256]
    mean_out_offset = pid_b * 256 + pid_oc
    tl.store(mean_out_ptr + mean_out_offset, mean_result)


@torch.fx.wrap
def fused_conv_view_mean(
    input: torch.Tensor,  # [batch, 512, 64, 64]
    weight: torch.Tensor, # [256, 512, 1, 1]
    bias: torch.Tensor,   # [256]
    mean_input: torch.Tensor, # [batch, 4096, 256]
):
    """
    Fused conv2d + view + mean operation
    """
    batch = input.shape[0]
    in_channels = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]
    out_channels = weight.shape[0]
    
    # Conv output shape: [batch, 256, 64, 64] -> view to [batch, 256, 4096]
    conv_output = torch.empty((batch, out_channels, height * width), 
                              dtype=input.dtype, device=input.device)
    
    # Mean output shape: [batch, 1, 256]
    mean_output = torch.empty((batch, 1, out_channels),
                              dtype=mean_input.dtype, device=mean_input.device)
    
    # Grid: (batch, out_channels)
    grid = (batch, out_channels)
    
    fused_conv_view_mean_kernel[grid](
        input,
        weight,
        bias,
        mean_input,
        conv_output,
        mean_output,
        batch,
        in_channels,
        out_channels,
        height,
        width,
    )
    
    return conv_output, mean_output


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: conv2d + view + mean
    This matches the full computation from the model.
    """
    # Conv2d
    tmp_0 = in_0  # bias
    tmp_1 = in_1  # weight
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    
    # View
    tmp_3 = tmp_2.view(1, 256, -1)  # Note: using 1 for batch (will be parameterized)
    
    # Mean
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    
    return tmp_4, tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_conv_view_mean