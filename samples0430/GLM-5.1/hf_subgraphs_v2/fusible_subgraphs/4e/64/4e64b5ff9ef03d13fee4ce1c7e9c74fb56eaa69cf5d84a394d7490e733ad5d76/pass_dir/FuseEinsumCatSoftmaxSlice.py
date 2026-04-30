import torch
import triton
import triton.language as tl

# Pattern matching function - mirrors the exact computation in model.py
def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Fused Triton kernel: einsum + cat + softmax + slice in one kernel
# Avoids materializing intermediate tensors (einsum result, cat result)
# Computes softmax over [in_0, einsum] without concatenating into a 128-element vector
# by handling the two halves separately (max, exp, sum, normalize)
@triton.jit
def fused_einsum_cat_softmax_slice_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_full_ptr, out_slice_ptr,
    B, H, W, J,
    stride_in0_b, stride_in0_h, stride_in0_w, stride_in0_j,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_j,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_full_b, stride_out_full_h, stride_out_full_w, stride_out_full_j,
    stride_out_slice_b, stride_out_slice_h, stride_out_slice_w, stride_out_slice_j,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    # Each program handles one (b, h, w) position
    pid = tl.program_id(0)
    w = pid % W
    hw = pid // W
    h = hw % H
    b = hw // H
    
    j_offsets = tl.arange(0, BLOCK_J)
    j_mask = j_offsets < J
    
    # Step 1: Compute einsum result[b,h,w,j] = sum_c(in_2[b,c,h,w] * in_1[b,c,h,j])
    # This is a vector-matrix multiply: in_2[b,:,h,w]^T @ in_1[b,:,h,:] = [J] output
    einsum_acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    
    in1_bh_base = b * stride_in1_b + h * stride_in1_h
    in2_bhw_base = b * stride_in2_b + h * stride_in2_h + w * stride_in2_w
    
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        
        # Load in_2[b, c, h, w] - a vector over c dimension
        in2_vals = tl.load(in_2_ptr + in2_bhw_base + c_offsets * stride_in2_c, mask=c_mask, other=0.0).to(tl.float32)
        
        # Load in_1[b, c, h, j] - a [BLOCK_C, BLOCK_J] tile
        in1_ptrs = in_1_ptr + in1_bh_base + c_offsets[:, None] * stride_in1_c + j_offsets[None, :] * stride_in1_j
        in1_vals = tl.load(in1_ptrs, mask=c_mask[:, None] & j_mask[None, :], other=0.0).to(tl.float32)
        
        # Accumulate: einsum[j] += sum_c(in2[c] * in1[c, j])
        einsum_acc += tl.sum(in2_vals[:, None] * in1_vals, axis=0)
    
    # Step 2: Load in_0[b, h, w, j] - first half of the cat dimension
    in0_bhw_base = b * stride_in0_b + h * stride_in0_h + w * stride_in0_w
    in0_vals = tl.load(in_0_ptr + in0_bhw_base + j_offsets * stride_in0_j, mask=j_mask, other=0.0).to(tl.float32)
    
    # Step 3: Compute softmax over [in_0_vals, einsum_acc] (128 elements)
    # Instead of concatenating into a 128-element vector, handle two halves separately
    # This is mathematically equivalent to softmax(cat([in_0, einsum]))
    
    # Find global max across both halves for numerical stability
    max_in0 = tl.max(in0_vals, axis=0)
    max_einsum = tl.max(einsum_acc, axis=0)
    max_val = tl.maximum(max_in0, max_einsum)
    
    # Compute exp for both halves
    exp_in0 = tl.exp(in0_vals - max_val)
    exp_einsum = tl.exp(einsum_acc - max_val)
    
    # Sum all exp values
    sum_exp = tl.sum(exp_in0, axis=0) + tl.sum(exp_einsum, axis=0)
    
    # Normalize to get softmax values
    softmax_in0 = exp_in0 / sum_exp      # First J elements (indices 0..J-1)
    softmax_einsum = exp_einsum / sum_exp  # Last J elements (indices J..2J-1)
    
    # Step 4: Store results
    # out_slice = softmax[..., :64] = softmax_in0 (first half)
    out_slice_bhw_base = b * stride_out_slice_b + h * stride_out_slice_h + w * stride_out_slice_w
    tl.store(out_slice_ptr + out_slice_bhw_base + j_offsets * stride_out_slice_j, softmax_in0, mask=j_mask)
    
    # out_full = full softmax result (128 elements)
    out_full_bhw_base = b * stride_out_full_b + h * stride_out_full_h + w * stride_out_full_w
    # Store first half (indices 0..J-1)
    tl.store(out_full_ptr + out_full_bhw_base + j_offsets * stride_out_full_j, softmax_in0, mask=j_mask)
    # Store second half (indices J..2J-1)
    tl.store(out_full_ptr + out_full_bhw_base + (j_offsets + J) * stride_out_full_j, softmax_einsum, mask=j_mask)

# Kernel wrapper - decorated with @torch.fx.wrap for FX compatibility
@torch.fx.wrap
def fused_einsum_cat_softmax_slice(in_0, in_1, in_2):
    """
    Fused implementation of:
    einsum('bchw,bchj->bhwj', in_2, in_1) + cat([in_0, einsum], dim=-1) + softmax + slice[..., :64]
    
    in_0: [B, H, W, J] - energy/prior (first half of softmax input)
    in_1: [B, C, H, J] - key tensor (einsum second operand)
    in_2: [B, C, H, W] - query tensor (einsum first operand)
    
    Returns: (out_full [B, H, W, 2J], out_slice [B, H, W, J])
    """
    # Derive dimensions from tensor shapes
    # einsum 'bchw,bchj->bhwj': in_2=[B,C,H,W], in_1=[B,C,H,J]
    B = in_0.shape[0]
    C = in_1.shape[1]  # C dimension from einsum pattern
    H = in_0.shape[1]  # H dimension (shared across all tensors)
    W = in_0.shape[2]  # W dimension (matches einsum output and in_0)
    J = in_0.shape[3]  # J dimension (last dim of in_0, matches einsum output last dim)
    
    cat_dim_size = J * 2  # Concatenated dimension size (128)
    
    # Allocate output tensors
    out_full = torch.empty(B, H, W, cat_dim_size, dtype=in_0.dtype, device=in_0.device)
    out_slice = torch.empty(B, H, W, J, dtype=in_0.dtype, device=in_0.device)
    
    # Kernel configuration
    BLOCK_C = 16  # Tile size for C dimension reduction
    BLOCK_J = J   # Tile size for J dimension (covers full J=64)
    
    # Grid: one program per (b, h, w) position
    grid = (B * H * W,)
    
    fused_einsum_cat_softmax_slice_kernel[grid](
        in_0, in_1, in_2,
        out_full, out_slice,
        B, H, W, J,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out_full.stride(0), out_full.stride(1), out_full.stride(2), out_full.stride(3),
        out_slice.stride(0), out_slice.stride(1), out_slice.stride(2), out_slice.stride(3),
        C=C,
        BLOCK_C=BLOCK_C,
        BLOCK_J=BLOCK_J,
    )
    
    return (out_full, out_slice)

# Replacement function - returns the kernel wrapper function reference
def replacement_func():
    return fused_einsum_cat_softmax_slice