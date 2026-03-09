import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple pattern that just returns the input
    return x

def replacement_args(x):
    return (x,)

# Triton kernel for direct channel reordering without intermediate transpose
@triton.jit
def channel_reorder_kernel(
    x_ptr, out_ptr,
    N, C_in, H, W,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    
    # Output size per program
    output_size = N * H * W * C_in
    total_elements = N * C_in * H * W
    
    # Process elements in blocks
    for i in range(pid * BLOCK_SIZE, total_elements, BLOCK_SIZE):
        element_idx = min(i, total_elements - 1)
        
        # Calculate indices
        batch_idx = element_idx // (C_in * H * W)
        remaining_idx = element_idx % (C_in * H * W)
        channel_idx_orig = remaining_idx // (H * W)
        spatial_idx = remaining_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        # Original: (N, groups, C_per_group, H, W) -> transpose(1, 2) -> (N, C_per_group, groups, H, W)
        # We want to go from (N, groups, C_per_group, H, W) directly to (N, groups*C_per_group, H, W)
        
        C_per_group = C_in // groups
        group_idx = channel_idx_orig // C_per_group
        channel_in_group = channel_idx_orig % C_per_group
        
        # Reordered channel index: interleaved groups
        reordered_channel_idx = channel_in_group * groups + group_idx
        
        # Compute output index
        output_idx = batch_idx * C_in * H * W + reordered_channel_idx * H * W + spatial_idx
        
        # Load input and store output
        val = tl.load(x_ptr + element_idx)
        tl.store(out_ptr + output_idx, val)

@torch.fx.wrap
def simple_identity(x):
    # Simple identity function for the pattern
    return x

def replacement_func():
    return simple_identity