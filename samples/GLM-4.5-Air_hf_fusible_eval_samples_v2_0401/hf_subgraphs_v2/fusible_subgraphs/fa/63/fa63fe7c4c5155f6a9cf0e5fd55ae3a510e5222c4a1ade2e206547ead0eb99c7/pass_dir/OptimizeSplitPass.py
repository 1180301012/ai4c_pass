import torch
import triton
import triton.language as tl
import math

@triton.jit
def optimized_split_kernel(
    input_ptr, output1_ptr, output2_ptr, output3_ptr,
    batch_size: tl.constexpr, heads: tl.constexpr,
    feat_dim: tl.constexpr, height: tl.constexpr, width: tl.constexpr,
    split_sizes_1: tl.constexpr, split_sizes_2: tl.constexpr, split_sizes_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate total elements
    total_elements = batch_size * heads * feat_dim * height * width
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    # Process each element
    for idx in range(start_idx, end_idx):
        # Calculate indices
        batch = idx // (heads * feat_dim * height * width)
        remainder = idx % (heads * feat_dim * height * width)
        head = remainder // (feat_dim * height * width)
        remainder = remainder % (feat_dim * height * width)
        feat = remainder // (height * width)
        remainder = remainder % (height * width)
        h = remainder // width
        w = remainder % width
        
        # Determine which split this feature belongs to
        base_feat = 0
        output_ptr = None
        split_size = 0
        
        if feat < split_sizes_1:
            # First split
            output_ptr = output1_ptr
            split_size = split_sizes_1
            local_feat = feat
        elif feat < split_sizes_1 + split_sizes_2:
            # Second split
            output_ptr = output2_ptr
            split_size = split_sizes_2
            local_feat = feat - split_sizes_1
        else:
            # Third split
            output_ptr = output3_ptr
            split_size = split_sizes_3
            local_feat = feat - split_sizes_1 - split_sizes_2
        
        # Calculate output offset
        offset = (
            batch * heads * split_size * height * width +
            head * split_size * height * width +
            local_feat * height * width +
            h * width +
            w
        )
        
        # Load input and store to appropriate output
        input_offset = (
            batch * heads * feat_dim * height * width +
            head * feat_dim * height * width +
            feat * height * width +
            h * width +
            w
        )
        
        value = tl.load(input_ptr + input_offset, mask=(feat < feat_dim) & (h < height) & (w < width), other=0.0)
        tl.store(output_ptr + offset, value)

@torch.fx.wrap
def optimized_split(input_tensor, split_sizes, dim=1):
    """
    Optimized split operation that writes to three output tensors directly
    """
    batch_size, heads, feat_dim, height, width = input_tensor.shape
    split_sizes_1, split_sizes_2, split_sizes_3 = split_sizes
    
    # Verify split sizes
    assert split_sizes_1 + split_sizes_2 + split_sizes_3 == feat_dim, "Invalid split sizes"
    
    # Create output tensors
    output1 = torch.empty(batch_size, heads, split_sizes_1, height, width, 
                         dtype=input_tensor.dtype, device=input_tensor.device)
    output2 = torch.empty(batch_size, heads, split_sizes_2, height, width, 
                         dtype=input_tensor.dtype, device=input_tensor.device)
    output3 = torch.empty(batch_size, heads, split_sizes_3, height, width, 
                         dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set kernel configuration
    BLOCK_SIZE = 1024
    
    # Calculate grid size using regular Python math
    total_elements = batch_size * heads * feat_dim * height * width
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_split_kernel[grid](
        input_tensor, output1, output2, output3,
        batch_size, heads, feat_dim, height, width,
        split_sizes_1, split_sizes_2, split_sizes_3,
        BLOCK_SIZE
    )
    
    # Return as tuple to match original pattern
    return (output1, output2, output3)

def pattern(x):
    sizes = [38, 57, 57]  # Default sizes for pattern matching
    split_result = torch.functional.split(x, sizes, dim=1)
    return split_result

def replacement_args(x):
    return (x, [38, 57, 57])  # Default sizes

def replacement_func():
    return optimized_split