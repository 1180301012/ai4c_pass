import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.nn.functional.interpolate(in_1, size=(32, 24), mode='nearest')
    tmp_1 = in_2 * tmp_0
    tmp_2 = torch.nn.functional.interpolate(in_0, size=(16, 12), mode='nearest')
    tmp_3 = in_3 * tmp_2
    return (tmp_1, tmp_3)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def interpolate_multiply_kernel_1(
    x_ptr, scale_ptr, out_ptr,
    n_channels, height_out, width_out, batch_size,
    scale_height: tl.constexpr, scale_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # First interpolate-multiply operation (in_1 -> (32,24) * in_2)
    pid = tl.program_id(0)
    block_idx = pid * BLOCK_SIZE
    offsets = block_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * n_channels * height_out * width_out)
    
    # Load input data with coordinate calculation
    idx = offsets
    batch = idx // (n_channels * height_out * width_out)
    channel = (idx % (n_channels * height_out * width_out)) // (height_out * width_out)
    h_out = (idx % (height_out * width_out)) // width_out
    w_out = idx % width_out
    
    # Calculate input coordinates for nearest neighbor interpolation
    h_in = (h_out * scale_height) // 24 if scale_height > 0 else h_out
    w_in = (w_out * scale_width) // 24 if scale_width > 0 else w_out
    h_in = tl.minimum(h_in, 15)  # Clamp to original height (16-1)
    w_in = tl.minimum(w_in, 11)  # Clamp to original width (12-1)
    
    # Load input and scale
    in_idx = (batch * n_channels + channel) * 16 * 12 + h_in * 12 + w_in
    x = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
    
    # Compute output
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def interpolate_multiply_kernel_2(
    x_ptr, scale_ptr, out_ptr,
    n_channels, height_out, width_out, batch_size,
    scale_height: tl.constexpr, scale_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Second interpolate-multiply operation (in_0 -> (16,12) * in_3)
    pid = tl.program_id(0)
    block_idx = pid * BLOCK_SIZE
    offsets = block_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * n_channels * height_out * width_out)
    
    # Load input data
    idx = offsets
    batch = idx // (n_channels * height_out * width_out)
    channel = (idx % (n_channels * height_out * width_out)) // (height_out * width_out)
    h_out = (idx % (height_out * width_out)) // width_out
    w_out = idx % width_out
    
    # Calculate input coordinates for nearest neighbor interpolation
    h_in = (h_out * scale_height) // 12 if scale_height > 0 else h_out
    w_in = (w_out * scale_width) // 12 if scale_width > 0 else w_out
    h_in = tl.minimum(h_in, 11)  # Clamp to original height (12-1)
    w_in = tl.minimum(w_in, 5)   # Clamp to original width (6-1)
    
    # Load input and scale
    in_idx = (batch * n_channels + channel) * 8 * 6 + h_in * 6 + w_in
    x = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
    
    # Compute output
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_interpolate_multiply(in_0, in_1, in_2, in_3):
    # Handle first operation: interpolate in_1 to (32,24) and multiply by in_2
    B0, C0, H0_original, W0_original = in_1.shape
    B0, C0_out, H0_out, W0_out = in_2.shape
    
    out1 = torch.empty((B0, C0_out, H0_out, W0_out), device=in_1.device, dtype=in_1.dtype)
    
    BLOCK_SIZE = 1024
    total_elements = B0 * C0_out * H0_out * W0_out
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    interpolate_multiply_kernel_1[(num_programs,)](
        in_1, in_2, out1,
        C0_out, H0_out, W0_out, B0,
        H0_original, W0_original,
        BLOCK_SIZE
    )
    
    # Handle second operation: interpolate in_0 to (16,12) and multiply by in_3
    B1, C1, H1_original, W1_original = in_0.shape
    B1, C1_out, H1_out, W1_out = in_3.shape
    
    out2 = torch.empty((B1, C1_out, H1_out, W1_out), device=in_0.device, dtype=in_0.dtype)
    
    total_elements = B1 * C1_out * H1_out * W1_out
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    interpolate_multiply_kernel_2[(num_programs,)](
        in_0, in_3, out2,
        C1_out, H1_out, W1_out, B1,
        H1_original, W1_original,
        BLOCK_SIZE
    )
    
    return out1, out2

def replacement_func():
    return fused_interpolate_multiply