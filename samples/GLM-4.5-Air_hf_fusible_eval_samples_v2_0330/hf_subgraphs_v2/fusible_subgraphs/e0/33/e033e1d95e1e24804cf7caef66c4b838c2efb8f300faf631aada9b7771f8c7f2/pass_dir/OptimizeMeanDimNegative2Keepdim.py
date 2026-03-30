import torch
import triton
import triton.language as tl

def pattern(in_2):
    # Mean operation from the original computation
    tmp_4 = in_2.mean(dim = -2, keepdim = True)
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def optimized_mean_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    n_features: tl.constexpr,
    seq_len: tl.constexpr,
):
    # Calculate program ID - each program handles one batch element
    pid = tl.program_id(0)
    
    # Check bounds to prevent out-of-access
    if pid >= batch_size:
        return
    
    # Calculate output position for this batch
    out_pos = pid * n_features * 1  # keepdim=True, so output has shape [B, 1, n_features]
    
    # Load slice for this batch: [seq_len, n_features]
    input_slice_ptr = x_ptr + pid * seq_len * n_features
    input_slice = tl.load(
        input_slice_ptr + tl.arange(0, seq_len)[:, None] * n_features + tl.arange(0, n_features)[None, :],
        other=0.0
    )
    
    # Compute mean over the sequence dimension (dim=-2)
    sum_result = tl.sum(input_slice, axis=0)
    mean_result = sum_result / seq_len
    
    # Store result with keepdim=True: [1, n_features] for this batch
    out_slice_ptr = out_ptr + out_pos
    tl.store(
        out_slice_ptr + tl.arange(0, n_features)[None, :],
        mean_result,
    )

@torch.fx.wrap
def optimized_mean(in_2):
    B, seq_len, n_features = in_2.shape
    
    # Create output tensor with keepdim=True shape: [B, 1, n_features]
    out = torch.empty((B, 1, n_features), dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration: one program per batch element
    grid = (B,)
    
    # Calculate optimal block sizes
    BLOCK_SIZE_FEATURES = min(256, n_features)
    
    # Launch the kernel
    optimized_mean_kernel[grid](
        x_ptr=in_2,
        out_ptr=out,
        batch_size=B,
        n_features=n_features,
        seq_len=seq_len,
    )
    
    return out

def replacement_func():
    return optimized_mean