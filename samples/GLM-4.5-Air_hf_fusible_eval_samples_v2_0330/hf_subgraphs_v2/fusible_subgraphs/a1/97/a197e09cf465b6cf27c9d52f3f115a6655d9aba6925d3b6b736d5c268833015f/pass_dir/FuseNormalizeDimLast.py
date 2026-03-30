import torch
import triton
import triton.language as tl

@triton.jit
def normalize_kernel(
    x_ptr,
    sum_ptr,
    out_ptr,
    batch_c_h: tl.constexpr,
    w_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch-channel-height dimension
    pid = tl.program_id(0)
    
    # Calculate offset for this element
    offsets = pid * w_size + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < (batch_c_h * w_size)
    
    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load precomputed sum for this batch-channel-height position
    sum_val = tl.load(sum_ptr + pid)
    
    # Calculate normalization factor (1/sum)
    norm_factor = tl.where(tl.abs(sum_val) > 1e-6, 1.0 / sum_val, 0.0)
    
    # Apply normalization
    out = x * norm_factor
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_normalize(x):
    """Fused normalization: sum along last dimension and divide"""
    # Compute sum along last dimension using PyTorch (this is what we're optimizing)
    # x has shape [1, 16, 196, 196]
    sums = x.sum(dim=-1)  # Result shape: [1, 16, 196]
    
    # Create sum buffer for kernel - needs to be on same device
    sum_buffer = sums.contiguous()
    
    # Get original shape
    original_shape = x.shape
    batch_c_h = original_shape[0] * original_shape[1] * original_shape[2]  # 1*16*196 = 3136
    w_size = original_shape[3]  # 196
    
    # Flatten input for easier processing
    x_flat = x.contiguous().view(batch_c_h, w_size)  # [3136, 196]
    
    # Output buffer
    out_flat = torch.empty_like(x_flat)
    
    # Block size for processing
    BLOCK_SIZE = 1024
    
    # Launch kernel - one program per batch-c-h element
    normalize_kernel[(batch_c_h,)](
        x_ptr=x_flat,
        sum_ptr=sum_buffer,
        out_ptr=out_flat,
        batch_c_h=batch_c_h,
        w_size=w_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original
    return out_flat.view(original_shape)

def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    return (tmp_0,)

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return fused_normalize