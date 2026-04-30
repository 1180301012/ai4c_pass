import torch
import triton
import triton.language as tl


@triton.jit
def fused_view_softmax_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Reshape input to [num_groups, 1, group_size] format
    2. Softmax along dim=2 (the group_size dimension)
    3. Unsqueeze to add trailing dimension -> [num_groups, 1, group_size, 1]
    
    This kernel uses online softmax for numerical stability and efficiency.
    """
    # Calculate which group this program block handles
    pid = tl.program_id(0)
    
    # Each program handles one group (e.g., one batch element)
    # Input offset for this group
    group_offset_base = pid * group_size
    
    # Load all elements for this group
    offsets = group_offset_base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * group_size
    
    # Handle case where group_size might not be divisible by BLOCK_SIZE
    load_mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=load_mask, other=0.0)
    
    # Compute online softmax for numerical stability
    # Step 1: Find max value in the group
    block_max = tl.max(x, axis=0)
    
    # Step 2: Compute exp(x - max) and sum
    exp_vals = tl.exp(tl.where(load_mask, x - block_max, 0.0))
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Step 3: Normalize
    softmax_vals = exp_vals / sum_exp
    
    # Step 4: Store result with unsqueeze (add trailing dimension)
    # Output shape: [num_groups, 1, group_size, 1]
    # We store as [num_groups * group_size, 1] to match the view semantics
    output_offsets = group_offset_base + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < n_elements
    
    tl.store(output_ptr + output_offsets, softmax_vals, mask=output_mask)


@torch.fx.wrap
def fused_view_softmax_unsqueeze_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function that launches the fused kernel.
    
    Input shape: [N, 1, M] where N is num_groups, M is group_size
    Output shape: [N, 1, M, 1]
    """
    n_elements = x.numel()
    num_groups = x.shape[0]
    group_size = x.shape[2]
    
    # Allocate output tensor
    output = torch.empty((num_groups, 1, group_size, 1), 
                        dtype=x.dtype, 
                        device=x.device)
    
    # Define block size for efficient execution
    BLOCK_SIZE = 1024
    
    # Calculate number of programs (one per group)
    num_programs = num_groups
    
    # Launch kernel
    fused_view_softmax_unsqueeze_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        num_groups=num_groups,
        group_size=group_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: Conv2D -> View -> Softmax -> Unsqueeze
    
    This pattern matches the attention mechanism in models like:
    - S-ViPNAS-Res50 (attention modules)
    - GCNet_R101 (GC block attention)
    
    The Conv2D is a 1x1 convolution with bias, followed by reshape to 
    [N, 1, M], softmax along dim=2, and unsqueeze to add trailing dimension.
    """
    # Conv2D with 1x1 kernel, stride=1, padding=0, dilation=1, groups=1
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # View: reshape to [N, 1, M] format
    # The first dimension is the group count, second is always 1, third is sequence length
    tmp_3 = conv2d.view(conv2d.shape[0], 1, -1)
    
    # Softmax along dim=2 (the sequence dimension)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    
    # Unsqueeze: add trailing dimension
    tmp_5 = tmp_4.unsqueeze(-1)
    
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    We need the output of the Conv2D to compute the reshape parameters.
    """
    # Return the Conv2D output which will be computed by the pattern
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the replacement function that implements the fused kernel.
    """
    return fused_view_softmax_unsqueeze_wrapper