import torch
import triton
import triton.language as tl

# Pattern matching for scale factor expansion
def pattern(in_0, in_1, in_2):
    """
    Pattern matching for the tensor expansion operations that create tmp_10:
    tmp_9 = in_1.unsqueeze(-1)  
    tmp_10 = tmp_9.unsqueeze(-1)
    Returns (tmp_8, tmp_10) where tmp_8 comes from main computation and tmp_10 is the expanded result
    """
    # Match the entire computation but only optimize the expansion part
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1) 
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    tmp_9 = in_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return (tmp_8, tmp_10)

# Argument extraction for replacement
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_expansion_kernel(
    scale_ptr,    # in_1 - [C]
    output_ptr,   # tmp_10 - [C, 1, 1]
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    
    # Each program handles multiple scale factors
    scale_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = scale_ids < channels
    
    # Load scale factors
    scale_values = tl.load(scale_ptr + scale_ids, mask=scale_ids < channels, other=0.0)
    
    # Create expanded output: [C, 1, 1] 
    # Each scale value is broadcasted to a singleton tensor
    # In the output tensor, we flatten the [C, 1, 1] to [C] for simplicity
    # The calling function will handle the reshaping
    tl.store(output_ptr + scale_ids, scale_values, mask=scale_ids < channels)

@torch.fx.wrap
def optimized_expansion_wrapper(in_0, in_1, in_2):
    """Wrapper function to launch the optimized expansion kernel"""
    channels = in_1.shape[0]
    
    # Create output tensor with shape [channels, 1, 1]
    output = torch.empty((channels, 1, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Flatten output for kernel processing
    output_flat = output.view(-1)  # [channels]
    
    # Calculate grid dimensions
    BLOCK_SIZE = 256  # Optimal block size for this operation
    num_programs = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_expansion_kernel[(num_programs,)](
        scale_ptr=in_1,
        output_ptr=output_flat,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns kernel wrapper)  
def replacement_func():
    return optimized_expansion_wrapper