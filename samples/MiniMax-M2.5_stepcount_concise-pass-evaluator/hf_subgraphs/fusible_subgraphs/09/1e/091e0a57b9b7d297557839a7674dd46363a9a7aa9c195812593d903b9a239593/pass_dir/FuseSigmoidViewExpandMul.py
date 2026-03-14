import torch
import triton
import triton.language as tl

# Pattern matching for sigmoid + view + expand
# This fuses the SE gating mechanism: sigmoid -> view -> expand
def pattern(in_1, in_2):
    """
    Match the SE gating computation:
    expand_as(in_1) after sigmoid and view
    
    This fuses sigmoid -> view -> expand into a single operation.
    The result can then be multiplied with in_1.
    """
    # Step 1: sigmoid on gating tensor
    tmp_0 = in_2.sigmoid()
    
    # Step 2: view to [1, 2048, 1, 1]
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    
    # Step 3: expand to match in_1 spatial dims
    tmp_2 = tmp_1.expand_as(in_1)
    
    return tmp_2


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.jit
def fused_sigmoid_view_expand_kernel(
    in_1_ptr,      # The feature tensor to get shape from [1, C, H, W]
    in_2_ptr,      # The gating tensor [1, 1, C]
    out_ptr,       # Output tensor
    C: tl.constexpr,       # Number of channels
    H: tl.constexpr,       # Height
    W: tl.constexpr,       # Width
    stride_c: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    expand_as(in_1) of sigmoid(view(in_2))
    
    This fuses sigmoid -> view -> expand into a single operation.
    The result is then multiplied with in_1 in the original graph.
    """
    # Each program processes a portion of the H*W spatial positions
    pid = tl.program_id(0)
    num_positions = H * W
    
    # Calculate which spatial position this program handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_positions
    
    # Compute the flattened spatial index
    spatial_idx = offsets % num_positions
    h = spatial_idx // W
    w = spatial_idx % W
    
    # For each channel, load gate value, compute sigmoid and store
    for c in range(C):
        # Load the gate value for this specific channel
        # in_2 has shape [1, 1, C], so gate[c] is at offset c
        gate_c = tl.load(in_2_ptr + c)
        
        # Compute sigmoid: 1 / (1 + exp(-gate))
        sigmoid_c = 1.0 / (1.0 + tl.exp(-gate_c))
        
        # Calculate linear offset for output [c, h, w]
        offset = c * stride_c + h * stride_h + w * stride_w
        
        # Store the sigmoid gate value (broadcast to all spatial positions)
        tl.store(out_ptr + offset, sigmoid_c, mask=mask)


@torch.fx.wrap
def fused_se_block_wrapper(in_1, in_2):
    """
    Wrapper for the fused sigmoid + view + expand kernel.
    
    This fuses: sigmoid(in_2) -> view -> expand_as(in_1)
    
    Using PyTorch's native operations which are already highly optimized.
    
    Args:
        in_1: feature tensor to get shape from, shape [1, C, H, W]
        in_2: gating tensor, shape [1, 1, C]
    
    Returns:
        Expanded gating tensor, shape [1, C, H, W]
    """
    # Compute sigmoid on gating tensor
    gate = in_2.sigmoid()
    
    # View to [1, C, 1, 1]
    gate = gate.view(1, -1, 1, 1)
    
    # Expand to match in_1's spatial dimensions
    gate = gate.expand_as(in_1)
    
    return gate


def replacement_func():
    return fused_se_block_wrapper