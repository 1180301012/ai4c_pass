import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_add_slice_cat_kernel(
    in0_ptr, in1_ptr, in2_ptr, out_ptr,
    in0_stride0, in0_stride1, in0_stride2, in0_stride3,
    in1_stride0, in1_stride1, in1_stride2, in1_stride3,
    in2_stride0, in2_stride1, in2_stride2, in2_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    slice_start,
    N, C_add, C_slice, H, W
):
    """
    Fused kernel for:
    tmp_0 = in_a + in_b  (element-wise add)
    tmp_1 = in_c[:, slice_start:, :, :]  (slice)
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    
    Each program handles one spatial position (b, h, w) and all channels.
    """
    pid = tl.program_id(0)
    
    # Calculate b, h, w from flat index
    # Each program processes one (b, h, w) position
    b = pid // (H * W)
    hw = pid % (H * W)
    h = hw // W
    w = hw % W
    
    # Process all channels for the add operation
    # C_add = number of channels to add
    for c in tl.range(C_add):
        # Load from in0
        in0_offset = b * in0_stride0 + c * in0_stride1 + h * in0_stride2 + w * in0_stride3
        x0 = tl.load(in0_ptr + in0_offset)
        
        # Load from in1
        in1_offset = b * in1_stride0 + c * in1_stride1 + h * in1_stride2 + w * in1_stride3
        x1 = tl.load(in1_ptr + in1_offset)
        
        # Compute add result
        add_result = x0 + x1
        
        # Store to output at channel c
        out_offset = b * out_stride0 + c * out_stride1 + h * out_stride2 + w * out_stride3
        tl.store(out_ptr + out_offset, add_result)
    
    # Process all channels for the slice operation
    # C_slice = number of channels to slice (C2 - slice_start)
    for c in tl.range(C_slice):
        # Load from in2 at channel slice_start + c
        src_c = slice_start + c
        in2_offset = b * in2_stride0 + src_c * in2_stride1 + h * in2_stride2 + w * in2_stride3
        x2 = tl.load(in2_ptr + in2_offset)
        
        # Store to output at channel slice_start + c
        dst_c = slice_start + c
        out_offset = b * out_stride0 + dst_c * out_stride1 + h * out_stride2 + w * out_stride3
        tl.store(out_ptr + out_offset, x2)


@torch.fx.wrap
def fused_add_slice_cat_wrapper(in0, in1, in2, add_variant=0):
    """
    Wrapper function for the fused kernel.
    
    add_variant=0: tmp_0 = in_0 + in_1; tmp_1 = in_2[:, X:] where X = in_0 channels
    add_variant=1: tmp_0 = in_0 + in_2; tmp_1 = in_1[:, X:] where X = in_0 channels
    """
    if add_variant == 0:
        # in0 + in1, then cat with sliced in2
        C0 = in0.shape[1]
        C1 = in1.shape[1]
        C2 = in2.shape[1]
        slice_start = C0
        C_slice = C2 - slice_start  # number of channels to slice
        
        # Output shape: [B, C2, H, W]
        B, C0_check, H, W = in0.shape
        assert C0_check == C0
        out = torch.empty(B, C2, H, W, device=in0.device, dtype=in0.dtype)
        
        # Grid: one program per spatial position
        N = B * H * W
        num_programs = N
        
        fused_add_slice_cat_kernel[(num_programs,)](
            in0_ptr=in0, in1_ptr=in1, in2_ptr=in2, out_ptr=out,
            in0_stride0=in0.stride(0), in0_stride1=in0.stride(1), in0_stride2=in0.stride(2), in0_stride3=in0.stride(3),
            in1_stride0=in1.stride(0), in1_stride1=in1.stride(1), in1_stride2=in1.stride(2), in1_stride3=in1.stride(3),
            in2_stride0=in2.stride(0), in2_stride1=in2.stride(1), in2_stride2=in2.stride(2), in2_stride3=in2.stride(3),
            out_stride0=out.stride(0), out_stride1=out.stride(1), out_stride2=out.stride(2), out_stride3=out.stride(3),
            slice_start=slice_start,
            N=N, C_add=C0, C_slice=C_slice, H=H, W=W
        )
    else:
        # in0 + in2, then cat with sliced in1
        C0 = in0.shape[1]
        C1 = in1.shape[1]
        C2 = in2.shape[1]
        slice_start = C0
        C_slice = C1 - slice_start  # number of channels to slice
        
        # Output shape: [B, C1, H, W]
        B, C0_check, H, W = in0.shape
        assert C0_check == C0
        out = torch.empty(B, C1, H, W, device=in0.device, dtype=in0.dtype)
        
        # Grid: one program per spatial position
        N = B * H * W
        num_programs = N
        
        # For variant 1, we swap in1 and in2: add in0+in2, slice from in1
        fused_add_slice_cat_kernel[(num_programs,)](
            in0_ptr=in0, in1_ptr=in2, in2_ptr=in1, out_ptr=out,
            in0_stride0=in0.stride(0), in0_stride1=in0.stride(1), in0_stride2=in0.stride(2), in0_stride3=in0.stride(3),
            in1_stride0=in2.stride(0), in1_stride1=in2.stride(1), in1_stride2=in2.stride(2), in1_stride3=in2.stride(3),
            in2_stride0=in1.stride(0), in2_stride1=in1.stride(1), in2_stride2=in1.stride(2), in2_stride3=in1.stride(3),
            out_stride0=out.stride(0), out_stride1=out.stride(1), out_stride2=out.stride(2), out_stride3=out.stride(3),
            slice_start=slice_start,
            N=N, C_add=C0, C_slice=C_slice, H=H, W=W
        )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Pattern: Add in_0 + in_1, slice in_2 from channel N, and concatenate.
    
    This matches: tmp_0 = in_0 + in_1; tmp_1 = in_2[:, N:]; tmp_2 = cat([tmp_0, tmp_1], dim=1)
    
    The slice position N must equal the channel dimension of in_0 (and in_1).
    """
    # First do the add
    tmp_add = in_0 + in_1
    
    # Get the slice position from the tensor dimension
    # This will be traced as a constant when using example inputs
    N = in_0.size(1)
    tmp_1 = in_2[:, N:]
    
    tmp_2 = torch.cat([tmp_add, tmp_1], dim=1)
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement function."""
    # The slice position should be the channel count of in_0
    slice_pos = in_0.shape[1]
    return (in_0, in_1, in_2, slice_pos)


def replacement_func():
    """Return the replacement function."""
    return fused_add_slice_cat_wrapper