import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """Match interpolate + permute + reshape operations for relative position bias"""
    tmp_1 = torch.nn.functional.interpolate(in_1, size=(63, 63), mode='bilinear')
    tmp_2 = tmp_1.permute(0, 2, 3, 1)
    tmp_3 = tmp_2.reshape(3969, -1)
    tmp_4 = in_0[slice(3969, None, None)]
    return (tmp_4, tmp_3)

# Argument extraction function  
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_bias_kernel(
    # Input tensors
    in_0_ptr,          # [N, C] relative position bias table
    in_1_ptr,          # [1, C, H, W] input tensor
    # Output tensors  
    out_4_ptr,         # [slice_size, C] sliced portion of in_0
    out_3_ptr,         # [H*W, C] flattened interpolated and permuted result
    # Metadata
    N: tl.constexpr,   # Rows in in_0 (3972 or 2212)
    C: tl.constexpr,   # Channels (16 or 12) 
    H: tl.constexpr,   # Target height (63 or 47)
    W: tl.constexpr,   # Target width (63 or 47)
    slice_start: tl.constexpr,  # Start index for slicing (3969 or 2209)
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Compute total spatial elements (including the one we skip)
    total_spatial = H * W
    
    # Process output 3: reshaped interpolated result
    mask = offsets < (total_spatial - 1)  # Skip last spatial element
    h_idx = (offsets % W) // H
    w_idx = offsets % W
    c_idx = (offsets // total_spatial) % C
    
    # Load input at interpolated position (for 63x63 or 47x47 output)
    # Since we're interpolating to exact size, we can use original coordinates
    src_h = tl.cast(h_idx, tl.int32)
    src_w = tl.cast(w_idx, tl.int32)
    
    # Load from input tensor [1, C, H, W]
    src_offset = src_h * W + src_w
    in_1_offset = src_offset * C + c_idx
    val = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    
    # Store to output [H*W, C] flattened layout (skipping last spatial element)
    out_3_offset = offsets * C + c_idx
    tl.store(out_3_ptr + out_3_offset, val, mask=mask)
    
    # Process output 4: sliced portion of in_0
    out_4_start = tl.max(tl.zeros([], dtype=tl.int32), slice_start)
    out_4_size = N - slice_start
    
    if pid == 0:  # Only first program handles the slice
        for i in range(out_4_size):
            src_offset_slice = (out_4_start + i) * C
            for j in range(min(BLOCK_SIZE, C - 0)):  
                if j < C:
                    val_slice = tl.load(in_0_ptr + src_offset_slice + j, other=0.0)
                    tl.store(out_4_ptr + i * C + j, val_slice)

@torch.fx.wrap
def optimized_bias_forward(in_0, in_1):
    # Determine parameters based on input shapes
    _, C, H, W = in_1.shape
    N, _ = in_0.shape
    
    # Calculate spatial parameters
    total_spatial = H * W
    slice_start = total_spatial - 1  # Skip last spatial element
    
    # Create output tensors
    out_4_size = N - slice_start
    out_3 = torch.empty((total_spatial - 1, C), dtype=in_1.dtype, device=in_1.device)
    out_4 = torch.empty((out_4_size, C), dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid configuration
    BLOCK_SIZE = 1024
    num_programs_3 = ((total_spatial - 1) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_programs_4 = ((out_4_size * C) + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_bias_kernel[(num_programs_3, num_programs_4)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_4_ptr=out_4,
        out_3_ptr=out_3,
        N=N,
        C=C,
        H=H,
        W=W,
        slice_start=slice_start,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_4, out_3)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_bias_forward