import torch
import triton
import triton.language as tl


def pattern(tmp_11):
    """
    Pattern to match: reshape followed by permute
    Input: tmp_11 with shape [729, 12]
    reshape(1, 27, 27, -1) produces [1, 27, 27, 12]
    permute(0, 3, 1, 2) gives [1, 12, 27, 27]
    """
    tmp_12 = tmp_11.reshape(1, 27, 27, -1)
    tmp_13 = tmp_12.permute(0, 3, 1, 2)
    return tmp_13


def replacement_args(tmp_11):
    return (tmp_11,)


@triton.jit
def reshape_permute_kernel(
    input_ptr, output_ptr,
    B, D, H, W,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused reshape(1, H, W, D) + permute(0, 3, 1, 2) kernel
    Input: [B*H*W, D] = [729, 12] where H=27, W=27, D=12, B=1
    Output: [B, D, H, W] = [1, 12, 27, 27]
    """
    # Get program id
    pid = tl.program_id(0)
    
    # Total output elements = B * D * H * W
    total_elements = B * D * H * W
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # For output [B, D, H, W]:
    b = offsets // (D * H * W)
    rem = offsets % (D * H * W)
    d = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W
    
    # Input is flat [B*H*W, D]
    # reshape(1, 27, 27, -1) treats it as [1, H, W, D] row-major
    # permute(0, 3, 1, 2) gives [1, D, H, W]
    # So output[b, d, h, w] = input[b*H*W + h*W + w, d]
    input_idx = b * H * W + h * W + w
    input_offsets = input_idx * D + d
    
    val = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Store to output [B, D, H, W]
    tl.store(output_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def reshape_permute_kernel_wrapper(x, output, B, D, H, W, num_programs, BLOCK_SIZE):
    """Wrapper to launch the Triton kernel"""
    reshape_permute_kernel[(num_programs,)](
        x, output,
        B, D, H, W,
        BLOCK_SIZE
    )
    return output


def reshape_permute_fused(x):
    """
    Fused reshape + permute operation.
    Input x: [729, 12]
    Output: [1, 12, 27, 27]
    """
    input_size, D = x.shape
    H = 27
    W = 27
    B = 1
    
    total_elements = B * D * H * W
    
    # Output shape: [1, 12, 27, 27]
    output = x.new_empty((B, D, H, W))
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    return reshape_permute_kernel_wrapper(x, output, B, D, H, W, num_programs, BLOCK_SIZE)


def replacement_func():
    return reshape_permute_fused