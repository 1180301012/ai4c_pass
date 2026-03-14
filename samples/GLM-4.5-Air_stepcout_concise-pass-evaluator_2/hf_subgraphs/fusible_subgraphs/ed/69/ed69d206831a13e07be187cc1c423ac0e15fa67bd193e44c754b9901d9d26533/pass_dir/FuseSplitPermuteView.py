import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact operations in model.py
def pattern(in_0, in_1):
    # Element-wise addition
    tmp_0 = in_1 + in_0
    
    # Split along dimension 1 
    tmp_1 = torch.functional.split(tmp_0, [1, -1], 1)
    
    # Get the second part of the split
    tmp_3 = tmp_1[1]
    
    # Permute last two dimensions
    tmp_4 = tmp_3.permute(0, 2, 1)
    
    # View/reshape to 4D tensor
    tmp_5 = tmp_4.view(1, 384, 14, 14)
    
    # Return both values that are observable outside the matched subgraph
    return tmp_1[0], tmp_5

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel to fuse split, permute, and view operations
@triton.jit
def fused_split_permute_view_kernel(
    input_ptr,
    output_0_ptr,  # First part of split (special token/prefix)
    output_1_ptr,  # Second part processed (spatial features)
    batch_size,
    seq_len_total,
    hidden_dim,
    spatial_dim1,  # 14 or 24
    spatial_dim2,  # 14 or 24
    split_offset: tl.constexpr,  # Where to split (after 1 element)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Process the second part of the split (sequence tokens)
    batch_idx = pid // (spatial_dim1 * spatial_dim2)
    spatial_1_idx = (pid % (spatial_dim1 * spatial_dim2)) // spatial_dim2
    spatial_2_idx = (pid % (spatial_dim1 * spatial_dim2)) % spatial_dim2
    
    # Calculate input indices
    input_offset = (batch_idx * seq_len_total + split_offset) * hidden_dim + spatial_1_idx * spatial_dim2 + spatial_2_idx
    
    # Load input data
    input_val = tl.load(input_ptr + input_offset, mask=input_offset < batch_size * (seq_len_total - split_offset) * hidden_dim, other=0.0)
    
    # Store processed output (permuted and viewed)
    output_1_offset = (batch_idx * hidden_dim * spatial_dim1 + spatial_1_idx) * spatial_dim2 + spatial_2_idx
    tl.store(output_1_ptr + output_1_offset, input_val)

# Kernel wrapper for the second part processing
@torch.fx.wrap
def fused_split_permute_view_kernel_wrapper(in_0, in_1):
    # Get input shapes
    batch_size = in_0.shape[0]
    seq_len_total = in_0.shape[1] 
    hidden_dim = in_0.shape[2]
    
    # Determine spatial dimensions based on sequence length
    # Total sequence after split is seq_len_total - 1 (remove the first token)
    spatial_total = seq_len_total - 1
    if spatial_total == 196:  # 14*14
        spatial_dim1, spatial_dim2 = 14, 14
    elif spatial_total == 576:  # 24*24  
        spatial_dim1, spatial_dim2 = 24, 24
    else:
        # Fallback for unknown dimensions - simple square root approximation
        spatial_dim1 = int((spatial_total ** 0.5))
        spatial_dim2 = spatial_total // spatial_dim1
    
    # Create output tensors
    output_0 = in_0[:, :1, :]  # First part of split (just clone the first element)
    output_1 = torch.empty((batch_size, hidden_dim, spatial_dim1, spatial_dim2), 
                          dtype=in_0.dtype, device=in_0.device)
    
    # Launch Triton kernel
    spatial_elements = spatial_dim1 * spatial_dim2
    batch_spatial_elements = batch_size * spatial_elements
    num_programs = (batch_spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_split_permute_view_kernel[(num_programs,)](
        in_1 + in_0,  # Element-wise addition is done here
        output_0,
        output_1,
        batch_size,
        seq_len_total, 
        hidden_dim,
        spatial_dim1,
        spatial_dim2,
        1,  # split_offset: keep first element, process from index 1
        BLOCK_SIZE=1024
    )
    
    return output_0, output_1

# Replacement function (returns the kernel wrapper)
def replacement_func():
    return fused_split_permute_view_kernel_wrapper