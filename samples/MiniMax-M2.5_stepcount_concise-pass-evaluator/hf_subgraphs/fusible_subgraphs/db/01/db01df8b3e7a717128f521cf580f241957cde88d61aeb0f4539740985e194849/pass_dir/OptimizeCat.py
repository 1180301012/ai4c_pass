import torch
import triton
import triton.language as tl

# Pattern matching: concatenation of 5 tensors along dim=1
def pattern(in_5, in_7, in_8, in_6, tmp_7):
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8

# Extract arguments needed for the replacement
def replacement_args(in_5, in_7, in_8, in_6, tmp_7):
    return (in_5, in_7, in_8, in_6, tmp_7)

# Optimized kernel: fused concatenation
@triton.jit
def concat_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr,
    out_ptr,
    H, W,
    c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr, c3: tl.constexpr, c4: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a 2D block of the output
    pid = tl.program_id(0)
    num_h = H
    num_w = W
    
    # Calculate h, w for this program
    h = pid // num_w
    w = pid % num_w
    
    if h >= H:
        return
    
    # Calculate output offset for this position
    base_offset = h * W + w
    
    # Load and store each input channel contribution
    # Output channels: [0:c0), [c0:c0+c1), [c0+c1:c0+c1+c2), [c0+c1+c2:c0+c1+c2+c3), [c0+c1+c2+c3:c0+c1+c2+c3+c4)
    
    # Input 0: channels 0 to c0
    for c in range(c0):
        offset = c * H * W + base_offset
        val = tl.load(in_0_ptr + c * H * W + base_offset)
        tl.store(out_ptr + offset, val)
    
    # Input 1: channels c0 to c0+c1
    for c in range(c1):
        offset = (c0 + c) * H * W + base_offset
        val = tl.load(in_1_ptr + c * H * W + base_offset)
        tl.store(out_ptr + offset, val)
    
    # Input 2: channels c0+c1 to c0+c1+c2
    for c in range(c2):
        offset = (c0 + c1 + c) * H * W + base_offset
        val = tl.load(in_2_ptr + c * H * W + base_offset)
        tl.store(out_ptr + offset, val)
    
    # Input 3: channels c0+c1+c2 to c0+c1+c2+c3
    for c in range(c3):
        offset = (c0 + c1 + c2 + c) * H * W + base_offset
        val = tl.load(in_3_ptr + c * H * W + base_offset)
        tl.store(out_ptr + offset, val)
    
    # Input 4: channels c0+c1+c2+c3 to c0+c1+c2+c3+c4
    for c in range(c4):
        offset = (c0 + c1 + c2 + c3 + c) * H * W + base_offset
        val = tl.load(in_4_ptr + c * H * W + base_offset)
        tl.store(out_ptr + offset, val)


@torch.fx.wrap
def concat_kernel_wrapper(in_5, in_7, in_8, in_6, tmp_7):
    """
    Optimized concatenation along dim=1
    Inputs: [1, 2048, 64, 64], [1, 512, 64, 64], [1, 512, 64, 64], [1, 512, 64, 64], [1, 512, 64, 64]
    Output: [1, 4096, 64, 64]
    """
    # Get shapes
    batch, c0, H, W = in_5.shape
    c1 = in_7.shape[1]
    c2 = in_8.shape[1]
    c3 = in_6.shape[1]
    c4 = tmp_7.shape[1]
    
    total_c = c0 + c1 + c2 + c3 + c4
    
    # Allocate output
    output = torch.empty((batch, total_c, H, W), device=in_5.device, dtype=in_5.dtype)
    
    # Grid: one program per spatial position
    grid = (H * W,)
    
    concat_kernel[grid](
        in_5, in_7, in_8, in_6, tmp_7,
        output,
        H, W,
        c0, c1, c2, c3, c4,
        BLOCK_SIZE=1,
    )
    
    return output


def replacement_func():
    return concat_kernel_wrapper