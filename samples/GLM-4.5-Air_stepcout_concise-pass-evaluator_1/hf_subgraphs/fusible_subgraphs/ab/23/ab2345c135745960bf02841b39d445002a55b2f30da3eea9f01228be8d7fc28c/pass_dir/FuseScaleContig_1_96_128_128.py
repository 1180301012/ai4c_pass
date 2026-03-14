import torch
import triton
import triton.language as tl

# Pattern: Element-wise multiplication + contiguous fusion
def pattern(in_2, tmp_4):
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(in_2, tmp_4):
    return (in_2, tmp_4)

# Optimized kernel: Fused element-wise multiplication with contiguous memory
@triton.jit
def fused_scale_contig_kernel(
    x_ptr,      # in_2: [1, 96, 128, 128]
    scale_ptr,  # tmp_4: [1, 96, 1, 1] 
    out_ptr,    # output: [1, 96, 128, 128]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate linear indices for each element to determine which scale factor to use
    # For tensor [1, 96, 128, 128], index calculation:
    # total_offset = batch * (96*128*128) + channel * (128*128) + h * 128 + w
    
    # For each element, calculate channel index and apply scaling
    # Channel index = offset // (H * W), where H=128, W=128
    channel_indices = offsets // (128 * 128)
    
    # Load scale factors for all channels in this block
    scale_vals = tl.load(scale_ptr + channel_indices, mask=(channel_indices < 96), other=1.0)
    
    # Apply element-wise multiplication
    x = x * scale_vals
    
    # Store the result (already contiguous by design since we process linearly)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap  
def fused_scale_contig(in_2, tmp_4):
    # Calculate total number of elements in the feature map
    n_elements = in_2.numel()
    
    # Determine optimal block size (power of 2 for Triton)
    BLOCK_SIZE = 2048  # Adjusted for better GPU occupancy
    
    # Calculate number of program instances needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(in_2)
    
    # Launch the fused kernel
    fused_scale_contig_kernel[(num_programs,)](
        x_ptr=in_2,
        scale_ptr=tmp_4.view(-1),  # Flatten scale tensor to [96]
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_scale_contig