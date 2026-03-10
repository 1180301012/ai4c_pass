import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern 1: in_0 + in_1 with slice from in_2
    tmp_0 = in_0 + in_1
    tmp_1 = in_2[slice(None, None, None), slice(0, None, None)]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    # For pattern: in_0 + in_1 with slice from in_2
    # The slice start is always the number of channels in the first tensor being added
    slice_start = in_0.shape[1]  # in_0 and in_1 have same shape, in_2 has more channels
    return (in_0, in_1, in_2, slice_start)

@triton.jit
def fused_add_slice_concat_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    n_batch,
    channels1,
    channels2,
    height,
    width,
    slice_start: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one spatial location
    pid = tl.program_id(0)
    total_elements = n_batch * height * width
    
    # Calculate spatial offsets
    spatial_offset = pid * BLOCK_SIZE
    spatial_end = min((pid + 1) * BLOCK_SIZE, total_elements)
    
    # Reshape spatial offset to batch, height, width coordinates
    batch_idx = spatial_offset // (height * width)
    spatial_remainder = spatial_offset % (height * width)
    h_idx = spatial_remainder // width
    w_idx = spatial_remainder % width
    
    # Initialize output pointers with stride info
    out_offset = batch_idx * (channels1 + channels2) * height * width + h_idx * width + w_idx
    out_base = out_ptr + out_offset * 2  # *2 for float32
    
    # Process each channel location
    for c1 in range(0, channels1, 1):  # Process one channel at a time for simplicity
        # For in_0 + in_1 part
        offset1 = batch_idx * channels1 * height * width + c1 * height * width + h_idx * width + w_idx
        
        val0 = tl.load(in0_ptr + offset1, mask=True, other=0.0)
        val1 = tl.load(in1_ptr + offset1, mask=True, other=0.0)
        sum_val = val0 + val1
        
        output_offset = out_base + c1 * 2
        tl.store(output_offset, sum_val, mask=True)
    
    # For in_2 slice part  
    for c2 in range(slice_start, slice_start + channels2, 1):
        offset2 = batch_idx * channels2 * height * width + (c2 - slice_start) * height * width + h_idx * width + w_idx
        
        val2 = tl.load(in2_ptr + offset2, mask=True, other=0.0)
        
        output_offset = out_base + (channels1 + (c2 - slice_start)) * 2
        tl.store(output_offset, val2, mask=True)

@torch.fx.wrap
def fused_add_slice_concat(in_0, in_1, in_2, slice_start):
    n_batch, channels1, height, width = in_0.shape
    channels2 = in_2.shape[1] - slice_start  # Number of channels to slice
    
    # Calculate output shape
    out_channels = channels1 + channels2
    out_shape = (n_batch, out_channels, height, width)
    out = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    # Calculate grid size
    total_spatial_elements = n_batch * height * width
    BLOCK_SIZE = 1024  # Number of spatial locations per program
    num_programs = (total_spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate total channel elements for all programs
    total_elements = total_spatial_elements * out_channels
    
    # Launch kernel with 3D grid: (batch_spatial_programs, out_channels, 1)
    fused_add_slice_concat_kernel[(num_programs,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out_ptr=out,
        n_batch=n_batch,
        channels1=channels1,
        channels2=channels2,
        height=height,
        width=width,
        slice_start=slice_start,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_add_slice_concat