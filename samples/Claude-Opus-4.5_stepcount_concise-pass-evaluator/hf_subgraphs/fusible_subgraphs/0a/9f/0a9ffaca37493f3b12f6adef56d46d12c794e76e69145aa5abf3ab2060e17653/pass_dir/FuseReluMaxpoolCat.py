import torch
import triton
import triton.language as tl

# Pattern matching function - try just matching cat
def pattern(tmp_0, tmp_1, tmp_2, tmp_3):
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4

# Argument extraction function
def replacement_args(tmp_0, tmp_1, tmp_2, tmp_3):
    return (tmp_0, tmp_1, tmp_2, tmp_3)


# Triton kernel for concatenating 4 tensors along channel dimension
@triton.jit
def cat4_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total elements per input tensor
    total_in = N * C * HW
    
    # Process BLOCK_SIZE elements at a time
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_in
    
    # Load from all 4 inputs
    v0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    v1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    v2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    v3 = tl.load(in3_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate output positions for each input
    # Input offset breaks down as: batch*C*HW + channel*HW + spatial
    batch_idx = offsets // (C * HW)
    rem = offsets % (C * HW)
    channel_idx = rem // HW
    spatial_idx = rem % HW
    
    # Output has 4*C channels
    out_C = 4 * C
    out_base = batch_idx * out_C * HW
    
    # Store each input to its corresponding channel group
    out_off_0 = out_base + channel_idx * HW + spatial_idx
    out_off_1 = out_base + (C + channel_idx) * HW + spatial_idx
    out_off_2 = out_base + (2*C + channel_idx) * HW + spatial_idx
    out_off_3 = out_base + (3*C + channel_idx) * HW + spatial_idx
    
    tl.store(out_ptr + out_off_0, v0, mask=mask)
    tl.store(out_ptr + out_off_1, v1, mask=mask)
    tl.store(out_ptr + out_off_2, v2, mask=mask)
    tl.store(out_ptr + out_off_3, v3, mask=mask)


@torch.fx.wrap
def triton_cat4(tmp_0, tmp_1, tmp_2, tmp_3):
    # Get input shape - all inputs should have same shape
    N, C, H, W = tmp_0.shape
    HW = H * W
    
    # Allocate output tensor with 4*C channels
    out = torch.empty((N, 4 * C, H, W), dtype=tmp_0.dtype, device=tmp_0.device)
    
    # Launch kernel
    total_elements = N * C * HW
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_programs,)
    
    cat4_kernel[grid](
        tmp_0, tmp_1, tmp_2, tmp_3,
        out,
        N, C, HW,
        BLOCK_SIZE,
    )
    
    return out


# Replacement function returns the optimized function
def replacement_func():
    return triton_cat4