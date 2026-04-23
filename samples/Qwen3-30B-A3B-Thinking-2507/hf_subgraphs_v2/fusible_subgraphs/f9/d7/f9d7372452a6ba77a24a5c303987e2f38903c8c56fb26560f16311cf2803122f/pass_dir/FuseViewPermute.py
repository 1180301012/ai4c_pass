import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_1):
    # Match view followed by permute
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4

# Argument extraction function

def replacement_args(in_1):
    return (in_1,)

# Optimized kernel
@triton.jit

def view_permute_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr = 64,
    W: tl.constexpr = 48,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute the output index
    j = offsets // 32  # j: dimension 1 of output (flattened)
    k = offsets % 32    # k: dimension 2 of output (channels)
    
    # Convert to input index: (0, k, j//W, j%W)
    input_offset = k * (H * W) + (j // W) * W + (j % W)
    
    # Load and store
    value = tl.load(in_ptr + input_offset, mask=mask)
    tl.store(out_ptr + offsets, value, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap

def view_permute(in_1):
    # Calculate output shape: (1, 64*48, 32)
    N, C, H, W = in_1.shape
    out_shape = (N, H * W, C)
    
    out = torch.empty(out_shape, device=in_1.device, dtype=in_1.dtype)
    
    BLOCK_SIZE = 256
    num_programs = (H * W * C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    view_permute_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        n_elements=H * W * C,
        BLOCK_SIZE=BLOCK_SIZE,
        H=H,
        W=W,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return view_permute