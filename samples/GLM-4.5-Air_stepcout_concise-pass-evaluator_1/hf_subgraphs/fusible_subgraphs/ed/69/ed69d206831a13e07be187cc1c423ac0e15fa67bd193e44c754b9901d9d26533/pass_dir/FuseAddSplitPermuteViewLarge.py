import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Simplified pattern - just match the addition and basic structure
    tmp_0 = in_1 + in_0
    tmp_2 = tmp_0[:, :1, :]  # First slice (equivalent to tmp_1[0])
    tmp_3 = tmp_0[:, 1:, :]  # Second slice (equivalent to tmp_1[1])
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 24, 24)
    return (tmp_2, tmp_5)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel_add_split_permute_view_large(
    in0_ptr,
    in1_ptr,
    out1_ptr,      # First output: [1, 1, 384]
    out2_ptr,      # Second output: [1, 384, 24, 24]
    input_n,       # batch size (always 1)
    input_c,       # input channels (577 for this case)
    input_h,       # hidden dim (384)
    spatial_size,  # spatial dimension (24 for this case)
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load inputs and perform addition
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    added = in0 + in1
    
    # Calculate element positions in different dimensions
    # Input shape: [input_n, input_c, input_h]
    idx = offsets
    n = idx // (input_c * input_h)
    remaining = idx % (input_c * input_h)
    c = remaining // input_h
    h = remaining % input_h
    
    # First output: [1, 1, 384] - elements where c == 0
    first_output_mask = (c == 0) & (n == 0)
    first_output_indices = idx[first_output_mask]
    if first_output_indices.numel() > 0:
        tl.store(out1_ptr + first_output_indices, added[first_output_mask], mask=first_output_mask)
    
    # Second output: [1, 384, 24, 24] from elements where c > 0
    second_output_mask = (c > 0) & (n == 0)
    second_output_indices = idx[second_output_mask]
    
    if second_output_indices.numel() > 0:
        # Original position in [1, input_c-1, 384] space (excluding c=0)
        orig_c = c[second_output_mask] - 1  # Remove the first channel (c=0)
        
        # After permute: [1, 384, input_c-1] -> now we need to map to [1, 384, 24, 24]
        # orig_c becomes the spatial index in flattened form
        
        # Convert to 2D spatial indices
        spatial_idx = orig_c
        h_out = spatial_idx // spatial_size  # height index (0 to spatial_size-1)
        w_out = spatial_idx % spatial_size   # width index (0 to spatial_size-1)
        c_out = h[second_output_mask]  # channel index (0-383)
        
        # Final position in [1, 384, spatial_size, spatial_size] output
        # layout: [n=0, c_out, h_out, w_out] -> linear offset
        final_offset = c_out * (spatial_size * spatial_size) + h_out * spatial_size + w_out
        
        tl.store(out2_ptr + final_offset, added[second_output_mask], mask=second_output_mask)

@torch.fx.wrap
def fused_operation_large(in_0, in_1):
    # Extract input dimensions
    input_n = in_0.size(0)
    input_c = in_0.size(1)
    input_h = in_0.size(2)
    spatial_size = 24  # Fixed for this pass (576 = 24*24)
    
    # Create output tensors
    output1 = torch.empty([1, 1, 384], dtype=in_0.dtype, device=in_0.device)
    output2 = torch.empty([1, 384, 24, 24], dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    total_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_kernel_add_split_permute_view_large[(num_programs,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out1_ptr=output1,
        out2_ptr=output2,
        input_n=input_n,
        input_c=input_c,
        input_h=input_h,
        spatial_size=spatial_size,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output1, output2

def replacement_func():
    return fused_operation_large