import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation from model.py
def pattern(in_0, in_1):
    """Match the scale-subtract-split-squeeze-contiguous pattern"""
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return tmp_7, tmp_9

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel that fuses all operations
@triton.jit
def fused_scale_split_kernel(
    in_0_ptr,           # [1, 17, 1] int64 on CPU (will be converted on GPU)
    in_1_ptr,           # [1, 17, 2] float16/bfloat16 on GPU
    out_0_ptr,          # [1, 17, 1] output on GPU
    out_1_ptr,          # [1, 17, 1] output on GPU
    n_batch,            # batch size = 1
    n_seq,              # sequence length = 17  
    scale_factor,       # 1000000.0
    BLOCK_SIZE_M: tl.constexpr,  # block size for sequence dimension
):
    # Block index for sequence dimension
    m = tl.program_id(0)
    
    # Bounds checking
    if m >= n_seq:
        return
    
    # Load in_0 [1, 17, 1] and scale it [broadcast across last dimension]
    # in_0_ptr is [batch, n_seq, 1]
    idx_0 = m * 1
    in_0_val = tl.load(in_0_ptr + idx_0).to(tl.float32)
    scaled_0_val = in_0_val * scale_factor
    
    # Load in_1 [1, 17, 2] - both columns
    idx_1_base = m * 2
    in_1_val_0 = tl.load(in_1_ptr + idx_1_base)
    in_1_val_1 = tl.load(in_1_ptr + idx_1_base + 1)
    
    # Fused computation: subtraction and splitting
    # Column 0 (original first half)
    out_0_val = in_1_val_0 - scaled_0_val
    # Column 1 (original second half)  
    out_1_val = in_1_val_1 - scaled_0_val
    
    # Store both outputs
    tl.store(out_0_ptr + m, out_0_val)
    tl.store(out_1_ptr + m, out_1_val)

# Kernel wrapper that handles device transfers and data types
@torch.fx.wrap
def fused_scale_split_torch(in_0, in_1):
    # Move in_0 to GPU and convert to appropriate dtype
    device = in_1.device
    dtype = in_1.dtype
    
    # Transfer in_0 to GPU and convert to target dtype
    in_0_gpu = in_0.to(dtype).to(device)
    
    # Get input shapes
    batch, n_seq, _ = in_1.shape  # in_1 is [1, 17, 2]
    
    # Create output tensors on GPU
    out_0 = torch.empty((batch, n_seq), dtype=dtype, device=device)
    out_1 = torch.empty((batch, n_seq), dtype=dtype, device=device)
    
    # Launch Triton kernel
    grid = (n_seq,)  # One program per sequence position
    
    fused_scale_split_kernel[grid](
        in_0_gpu,
        in_1,
        out_0,
        out_1,
        batch,
        n_seq,
        1000000.0,
        BLOCK_SIZE_M=1  # Each thread handles one sequence position
    )
    
    return out_0, out_1

# Replacement function (returns the kernel wrapper function)
def replacement_func():
    return fused_scale_split_torch