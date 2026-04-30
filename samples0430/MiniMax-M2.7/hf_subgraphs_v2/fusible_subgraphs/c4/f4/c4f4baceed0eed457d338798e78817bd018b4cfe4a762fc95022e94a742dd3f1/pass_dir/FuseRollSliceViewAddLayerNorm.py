import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for: contiguous -> view -> roll -> slice -> contiguous -> view -> add -> layer_norm
    This pattern is common in Swin Transformer window operations.
    """
    # Ensure contiguous and reshape to 4D
    tmp_contiguous = in_3.contiguous()
    # Get original shape for dynamic dimension extraction
    orig_shape = in_3.shape
    
    # Infer dimensions from original shape
    # Shape is [1, A, 7, A, 7, C] -> reshape to [-1, A*7, A*7, C]
    H = orig_shape[1] * 7
    W = orig_shape[3] * 7
    C = orig_shape[5]
    
    tmp_4d = tmp_contiguous.view(-1, H, W, C)
    tmp_rolled = torch.roll(tmp_4d, shifts=(3, 3), dims=(1, 2))
    tmp_sliced = tmp_rolled[(slice(None, None, None), 
                             slice(None, H - 3, None), 
                             slice(None, W - 3, None), 
                             slice(None, None, None))]
    tmp_sliced_contig = tmp_sliced.contiguous()
    
    # Calculate output dimensions: (H-3) * (W-3) = new_seq_len
    new_seq_len = (H - 3) * (W - 3)
    tmp_3d = tmp_sliced_contig.view(1, new_seq_len, C)
    
    # Element-wise addition
    tmp_added = in_2 + tmp_3d
    
    # Layer normalization
    tmp_ln = torch.nn.functional.layer_norm(tmp_added, (C,), in_1, in_0, 1e-05)
    
    return tmp_added, tmp_ln


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2, in_3)


# Note: The first kernel is replaced by the cleaner v2 implementation below


# Redefined kernel with proper parallelization
@triton.jit
def fused_kernel_v2(
    # Input pointers
    in_3_ptr, in_2_ptr, weight_ptr, bias_ptr,
    # Output pointers  
    out_add_ptr, out_ln_ptr,
    # Strides for in_3 [1, A, 7, A, 7, C]
    in_3_s0, in_3_s1, in_3_s2, in_3_s3, in_3_s4, in_3_s5,
    # Strides for in_2 [1, seq, C]
    in_2_s0, in_2_s1, in_2_s2,
    # Output strides [1, seq, C]
    out_add_s0, out_add_s1, out_add_s2,
    out_ln_s0, out_ln_s1, out_ln_s2,
    # Dimensions
    orig_A, C,
    H, W, roll_shift,
    # Kernel config
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid: (out_seq_len, ) where out_seq_len = (H-3)*(W-3)
    Each program computes full layer norm for one sequence position.
    """
    pid = tl.program_id(0)
    
    out_seq_len = (H - 3) * (W - 3)
    
    if pid >= out_seq_len:
        return
    
    # Current sequence position
    r = pid // (W - 3)
    c = pid % (W - 3)
    
    # Apply roll: source at (r-3, c-3) with wrap-around
    # torch.roll shifts data by moving elements forward, so to find what ends up at position (r, c),
    # we need to look at (r - roll_shift) in the original
    rolled_r = (r - roll_shift + H) % H
    rolled_c = (c - roll_shift + W) % W
    
    # Convert to 6D indices
    a_idx = rolled_r // 7
    spatial_r = rolled_r % 7
    a_idx_c = rolled_c // 7
    spatial_c = rolled_c % 7
    
    # Compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    
    # Loop over channels
    for ch in range(C):
        # Load from in_3
        offset_in3 = (a_idx * in_3_s1 + spatial_r * in_3_s2 + 
                      a_idx_c * in_3_s3 + spatial_c * in_3_s4 + 
                      ch * in_3_s5)
        val = tl.load(in_3_ptr + offset_in3)
        
        # Load from in_2
        offset_in2 = pid * in_2_s1 + ch * in_2_s2
        val2 = tl.load(in_2_ptr + offset_in2)
        
        # Add
        val = val + val2
        
        # Store to out_add
        offset_add = pid * out_add_s1 + ch * out_add_s2
        tl.store(out_add_ptr + offset_add, val)
        
        # Accumulate for layer norm
        sum_val = sum_val + val
        sum_sq = sum_sq + val * val
    
    # Compute mean and variance
    mean = sum_val / C
    var = sum_sq / C - mean * mean
    std = tl.sqrt(var + 1e-05)
    
    # Normalize and store layer norm output
    for ch in range(C):
        # Load from out_add
        offset_add = pid * out_add_s1 + ch * out_add_s2
        val = tl.load(out_add_ptr + offset_add)
        
        # Normalize
        normalized = (val - mean) / std
        
        # Load weight and bias
        w = tl.load(weight_ptr + ch)
        b = tl.load(bias_ptr + ch)
        
        # Apply affine transform
        ln_val = normalized * w + b
        
        # Store to out_ln
        offset_ln = pid * out_ln_s1 + ch * out_ln_s2
        tl.store(out_ln_ptr + offset_ln, ln_val)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper for the fused kernel.
    Handles shape inference and kernel launch.
    """
    # Get dimensions
    orig_shape = in_3.shape  # [1, A, 7, A, 7, C]
    orig_A = orig_shape[1]
    C = orig_shape[5]
    
    # Compute output shape
    H = orig_A * 7  # A*7
    W = orig_A * 7
    out_seq_len = (H - 3) * (W - 3)  # (A*7-3)^2
    
    # Create output tensors
    out_add = torch.empty((1, out_seq_len, C), dtype=in_3.dtype, device=in_3.device)
    out_ln = torch.empty((1, out_seq_len, C), dtype=in_3.dtype, device=in_3.device)
    
    # Get strides
    in_3_s = in_3.stride()
    in_2_s = in_2.stride()
    out_add_s = out_add.stride()
    out_ln_s = out_ln.stride()
    
    # Launch configuration
    BLOCK_SIZE = 128
    num_programs = out_seq_len
    
    # Launch kernel
    fused_kernel_v2[(num_programs,)](
        in_3_ptr=in_3,
        in_2_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_add_ptr=out_add,
        out_ln_ptr=out_ln,
        # in_3 strides [1, A, 7, A, 7, C]
        in_3_s0=in_3_s[0], in_3_s1=in_3_s[1], in_3_s2=in_3_s[2],
        in_3_s3=in_3_s[3], in_3_s4=in_3_s[4], in_3_s5=in_3_s[5],
        # in_2 strides
        in_2_s0=in_2_s[0], in_2_s1=in_2_s[1], in_2_s2=in_2_s[2],
        # out_add strides
        out_add_s0=out_add_s[0], out_add_s1=out_add_s[1], out_add_s2=out_add_s[2],
        # out_ln strides
        out_ln_s0=out_ln_s[0], out_ln_s1=out_ln_s[1], out_ln_s2=out_ln_s[2],
        # Dimensions
        orig_A=orig_A,
        C=C,
        H=H, W=W,
        roll_shift=3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_add, out_ln


def replacement_func():
    """Return the replacement function."""
    return fused_kernel_wrapper