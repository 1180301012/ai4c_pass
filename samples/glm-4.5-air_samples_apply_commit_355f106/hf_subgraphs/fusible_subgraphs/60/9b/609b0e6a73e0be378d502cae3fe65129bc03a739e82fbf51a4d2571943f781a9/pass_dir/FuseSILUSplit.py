import torch

def pattern(a):
    return torch.nn.functional.silu(a, inplace=True)

def replacement_args(a):
    return (a,)

@triton.jit
def silu_split_kernel(
    in_ptr,
    out1_ptr,
    out2_ptr, 
    out3_ptr,
    batch_size,
    height,
    dim1_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel execution
    batch_idx = tl.program_id(0)
    dim1_idx = tl.program_id(1)
    
    # Load input data - apply SILU activation to entire row
    idx_off = tl.arange(0, BLOCK_SIZE)
    mask = idx_off < dim1_size
    
    # Calculate input offset for this (batch, height) pair
    in_offset = batch_idx * height * dim1_size + dim1_idx * dim1_size
    in_data = tl.load(in_ptr + in_offset + idx_off, mask=mask, other=0.0)
    
    # Apply SILU activation: x * sigmoid(x)
    silu_out = in_data / (1.0 + tl.exp(-in_data))
    
    # Store results in the three output segments for this (batch, height) pair
    # Segment 1: [0:512] - stored to out1
    mask1 = idx_off < 512
    out1_offset = batch_idx * height * 512 + dim1_idx * 512
    tl.store(out1_ptr + out1_offset + idx_off, silu_out, mask=mask1)
    
    # Segment 2: [512:1024] - stored to out2
    mask2 = (idx_off >= 512) & (idx_off < 1024)
    out2_offset = batch_idx * height * 512 + dim1_idx * 512
    tl.store(out2_ptr + out2_offset + idx_off - 512, silu_out, mask=mask2)
    
    # Segment 3: [1024:1152] - stored to out3
    mask3 = (idx_off >= 1024) & (idx_off < 1152)
    out3_offset = batch_idx * height * 128 + dim1_idx * 128
    tl.store(out3_ptr + out3_offset + idx_off - 1024, silu_out, mask=mask3)

@torch.fx.wrap
def fused_silu_split(in_1):
    # Get input dimensions
    batch_size, height, dim1_size = in_1.shape
    assert dim1_size == 1152, f"Expected dimension size 1152, got {dim1_size}"
    
    # Create output tensors
    out1 = torch.empty((batch_size, height, 512), dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((batch_size, height, 512), dtype=in_1.dtype, device=in_1.device)
    out3 = torch.empty((batch_size, height, 128), dtype=in_1.dtype, device=in_1.device)
    
    # Block size for Triton
    BLOCK_SIZE = 1024
    
    # Grid setup: (batch_size, height)
    grid = lambda meta: (batch_size, height)
    
    # Launch kernel
    silu_split_kernel[grid](
        in_ptr=in_1,
        out1_ptr=out1,
        out2_ptr=out2,
        out3_ptr=out3,
        batch_size=batch_size,
        height=height,
        dim1_size=dim1_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2, out3

def replacement_func():
    return fused_silu_split