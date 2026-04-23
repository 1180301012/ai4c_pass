import torch
import triton
import triton.language as tl

# Single program per channel - handle all 4096 spatial elements
# Use 512 programs total, each processing 4096 elements in a loop
BLOCK_SIZE = tl.constexpr(1024)

@triton.jit
def fused_se_block_kernel(
    in_0_ptr,       # Channel weights [1, 512]
    in_1_ptr,       # Input tensor [1, 512, 64, 64]
    out_ptr,        # Output tensor
    n_channels: tl.constexpr,
    n_spatial: tl.constexpr,
):
    """
    Fused Squeeze-and-Excitation block kernel.
    
    Grid: n_channels programs (one per channel)
    Each program loops through all spatial elements.
    
    Computes: relu(in_1 * (1 + sigmoid(in_0)))
    """
    # One program per channel
    channel_idx = tl.program_id(0)
    
    # Load channel weight
    weight = tl.load(in_0_ptr + channel_idx, mask=channel_idx < n_channels, other=0.0)
    
    # Compute sigmoid with clamped exp
    neg_weight = -weight
    neg_weight = tl.where(neg_weight > 20.0, 20.0, neg_weight)
    neg_weight = tl.where(neg_weight < -20.0, -20.0, neg_weight)
    neg_weight_fp32 = neg_weight.to(tl.float32)
    sigmoid_weight = (1.0 / (1.0 + tl.exp(neg_weight_fp32))).to(weight.dtype)
    
    # Scale factor
    scale_factor = 1.0 + sigmoid_weight
    
    # Base offset for this channel in the linearized tensor
    channel_base = channel_idx * n_spatial
    
    # Loop over spatial elements
    for start in range(0, n_spatial, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_spatial
        
        # Load
        val = tl.load(in_1_ptr + channel_base + offs, mask=mask, other=0.0)
        
        # Compute and apply ReLU
        scaled = val * scale_factor
        out_val = tl.where(scaled > 0, scaled, 0.0)
        
        # Store
        tl.store(out_ptr + channel_base + offs, out_val, mask=mask)


@torch.fx.wrap
def fused_se_block_wrapper(in_0: torch.Tensor, in_1: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the fused SE block kernel.
    
    Args:
        in_0: Channel weights tensor [1, 512]
        in_1: Input feature tensor [1, 512, H, W]
    
    Returns:
        Output tensor after SE block: relu(in_1 * (1 + sigmoid(in_0)))
    """
    # Get shapes
    n_channels = in_0.shape[1]  # 512
    B, C, H, W = in_1.shape
    n_spatial = H * W  # 64 * 64 = 4096
    
    # Allocate output
    out = torch.empty_like(in_1)
    
    # One program per channel
    grid = (n_channels,)
    
    fused_se_block_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_channels=n_channels,
        n_spatial=n_spatial,
    )
    
    return out


def pattern(in_0: torch.Tensor, in_1: torch.Tensor):
    """
    Match the SE block computation pattern:
    sigmoid -> view -> multiply -> add -> relu_
    
    This matches the pattern from model.py:
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    """
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    return tmp_4


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor):
    """
    Extract arguments for the replacement function.
    """
    return (in_0, in_1)


def replacement_func():
    """
    Return the fused kernel wrapper function.
    """
    return fused_se_block_wrapper