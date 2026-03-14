import torch
import triton
import triton.language as tl


def pattern(in_2):
    """
    Match the pattern: slice -> transpose -> reshape -> split
    This pattern appears in CoaT attention computation.
    
    Operations:
    1. Slice v tensor from index 1 onwards
    2. Transpose the sliced tensor
    3. Reshape to 4D tensor
    4. Split along the channel dimension
    
    The pattern returns the three split outputs (tmp_6, tmp_7, tmp_8)
    which appear in the model's return statement.
    """
    # Slice v from position 1 onwards
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    
    # Transpose last two dimensions
    tmp_3 = tmp_2.transpose(-1, -2)
    
    # Reshape to 4D tensor
    tmp_4 = tmp_3.reshape(1, 128, 96, 96)
    
    # Split along dim 1 - call split once and unpack
    split_result = tmp_4.split([32, 48, 48], dim=1)
    tmp_6 = split_result[0]
    tmp_7 = split_result[1]
    tmp_8 = split_result[2]
    
    # Return the split outputs that appear in the model's return
    return tmp_6, tmp_7, tmp_8


def replacement_args(in_2):
    """Extract arguments needed for the replacement function."""
    return (in_2,)


@triton.jit
def fused_transpose_reshape_split_kernel(
    input_ptr,
    output_ptr_0,
    output_ptr_1,
    output_ptr_2,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    split_0: tl.constexpr,
    split_1: tl.constexpr,
    split_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Slice from index 1 (skip first element)
    2. Transpose last two dimensions
    3. Reshape to [B, split_total, H', W']
    4. Split into 3 outputs
    
    Input shape: [B, H, S+1, D] (S+1 because we slice from 1)
    After slice: [B, H, S, D]
    After transpose: [B, H, D, S]
    After reshape: [B, split_total, H', W'] where H'*W' = S
    """
    # Calculate output spatial dimensions
    # S = H' * W'
    H_prime = tl.constexpr(96)
    W_prime = tl.constexpr(96)
    
    # Each program handles a portion of the output
    pid = tl.program_id(0)
    
    # Calculate total number of elements per output
    total_out_elements = split_0 * H_prime * W_prime
    
    # Determine which output this program handles
    num_outputs = 3
    outputs_per_program = (num_outputs + 2) // 3
    
    for out_idx in range(num_outputs):
        # Calculate output pointer offset
        if out_idx == 0:
            out_ptr = output_ptr_0
            split_size = split_0
        elif out_idx == 1:
            out_ptr = output_ptr_1
            split_size = split_1
        else:
            out_ptr = output_ptr_2
            split_size = split_2
        
        # Calculate offset for this output
        if out_idx == 0:
            offset = 0
        elif out_idx == 1:
            offset = split_0 * H_prime * W_prime
        else:
            offset = (split_0 + split_1) * H_prime * W_prime
        
        # Calculate element offset for this program
        elements_per_output = split_size * H_prime * W_prime
        program_offset = pid * BLOCK_SIZE
        
        # Process elements
        offs = tl.arange(0, BLOCK_SIZE)
        mask = program_offset + offs < elements_per_output
        
        # Calculate source indices for the fused operations
        # Input: [B, H, S+1, D], we slice from index 1 -> [B, H, S, D]
        # After transpose: [B, H, D, S]
        # After reshape: [B, split_total, H', W']
        
        # Global index into output
        global_idx = program_offset + offs
        
        # Calculate 4D indices in output [b, c, h, w]
        b = global_idx // (split_size * H_prime * W_prime)
        remainder = global_idx % (split_size * H_prime * W_prime)
        c = remainder // (H_prime * W_prime)
        remainder = remainder % (H_prime * W_prime)
        h = remainder // W_prime
        w = remainder % W_prime
        
        # Map to input indices after transpose and reshape
        # In reshape: output [b, c, h, w] corresponds to input [b, :, c, h*W'+w] after transpose
        # Since we have multiple heads H, we distribute c across heads
        c_adjusted = c * H // split_size
        h_in = h
        w_in = w
        
        # Input is [B, H, S+1, D], we want [B, H, D, S] after transpose
        # After slice: [B, H, S, D] (indices 1 to S)
        # So input index: [b, h, d, s] -> output [b, c, h, w]
        # d = c_adjusted, s = h_in * W' + w_in + 1 (slice from 1)
        
        d = c_adjusted
        s = h_in * W_prime + w_in + 1  # +1 because we skip index 0
        
        # Load from input [B, H, S+1, D] = [1, 8, 9217, 16]
        input_idx = b * H * (S + 1) * D + h * (S + 1) * D + s * D + d
        
        val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        tl.store(out_ptr + program_offset + offs, val, mask=mask)


@torch.fx.wrap
def fused_transpose_reshape_split_wrapper(in_2):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Input shapes:
    - in_2 (v): [1, 8, 9217, 16]
    
    Output shapes:
    - tmp_6: [1, 32, 96, 96]
    - tmp_7: [1, 48, 96, 96]
    - tmp_8: [1, 48, 96, 96]
    """
    B, H, S_plus_1, D = in_2.shape
    S = S_plus_1 - 1  # After slicing from index 1
    
    # Split sizes
    split_0, split_1, split_2 = 32, 48, 48
    
    # Output spatial dimensions (calculated from S = H' * W')
    # For S = 9216 = 96 * 96
    H_prime = 96
    W_prime = 96
    
    # Allocate outputs
    output_0 = torch.empty((B, split_0, H_prime, W_prime), dtype=in_2.dtype, device=in_2.device)
    output_1 = torch.empty((B, split_1, H_prime, W_prime), dtype=in_2.dtype, device=in_2.device)
    output_2 = torch.empty((B, split_2, H_prime, W_prime), dtype=in_2.dtype, device=in_2.device)
    
    # Configure kernel
    BLOCK_SIZE = 1024
    
    # Calculate grid
    total_elements_0 = split_0 * H_prime * W_prime
    total_elements_1 = split_1 * H_prime * W_prime
    total_elements_2 = split_2 * H_prime * W_prime
    max_elements = max(total_elements_0, total_elements_1, total_elements_2)
    num_programs = (max_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    grid = (num_programs,)
    
    fused_transpose_reshape_split_kernel[grid](
        in_2,
        output_0,
        output_1,
        output_2,
        B,
        H,
        S,
        D,
        split_0,
        split_1,
        split_2,
        BLOCK_SIZE,
    )
    
    return output_0, output_1, output_2


def replacement_func():
    """Return the replacement function."""
    return fused_transpose_reshape_split_wrapper