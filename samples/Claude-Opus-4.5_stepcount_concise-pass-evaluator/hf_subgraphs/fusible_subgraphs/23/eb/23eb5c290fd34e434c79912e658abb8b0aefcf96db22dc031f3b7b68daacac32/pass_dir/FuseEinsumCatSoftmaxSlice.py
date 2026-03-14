import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0, in_1, in_2):
    tmp_1 = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, tmp_1], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
    return (tmp_3, tmp_4)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_einsum_cat_softmax_kernel(
    in_0_ptr,      # [B, H, W, J] - energy tensor
    in_1_ptr,      # [B, C, H, J] - key tensor
    in_2_ptr,      # [B, C, H, W] - query tensor
    out_full_ptr,  # [B, H, W, 2*J] - full softmax output
    out_slice_ptr, # [B, H, W, J] - sliced output (first J elements)
    B, H, W, C, J,
    stride_in0_b, stride_in0_h, stride_in0_w, stride_in0_j,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_j,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_full_b, stride_out_full_h, stride_out_full_w, stride_out_full_j,
    stride_out_slice_b, stride_out_slice_h, stride_out_slice_w, stride_out_slice_j,
    BLOCK_C: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    # Each program handles one (b, h, w) position
    pid = tl.program_id(0)
    
    b = pid // (H * W)
    hw = pid % (H * W)
    h = hw // W
    w = hw % W
    
    # Create ranges for vectorized operations
    j_range = tl.arange(0, BLOCK_J)
    c_range = tl.arange(0, BLOCK_C)
    
    # Safety masks
    j_mask = j_range < J
    c_mask = c_range < C
    
    # Load query[b, :, h, w] - vector of size C
    # query: [B, C, H, W] with strides
    query_offsets = b * stride_in2_b + c_range * stride_in2_c + h * stride_in2_h + w * stride_in2_w
    query_vec = tl.load(in_2_ptr + query_offsets, mask=c_mask, other=0.0)
    
    # Load key[b, :, h, :] - matrix of size [C, J]
    # key: [B, C, H, J] with strides
    key_offsets = (b * stride_in1_b + 
                   c_range[:, None] * stride_in1_c + 
                   h * stride_in1_h + 
                   j_range[None, :] * stride_in1_j)
    key_mat = tl.load(in_1_ptr + key_offsets, mask=c_mask[:, None] & j_mask[None, :], other=0.0)
    
    # Compute einsum: output[j] = sum_c query[c] * key[c, j]
    # Result is a vector of size J
    einsum_result = tl.sum(query_vec[:, None] * key_mat, axis=0)
    
    # Load in_0[b, h, w, :] - vector of size J
    # in_0: [B, H, W, J] with strides
    in0_offsets = b * stride_in0_b + h * stride_in0_h + w * stride_in0_w + j_range * stride_in0_j
    in0_vec = tl.load(in_0_ptr + in0_offsets, mask=j_mask, other=0.0)
    
    # Compute softmax over concatenated [in0_vec, einsum_result] (size 2*J)
    # Numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    # Find max over both vectors
    max_in0 = tl.max(in0_vec)
    max_einsum = tl.max(einsum_result)
    max_val = tl.maximum(max_in0, max_einsum)
    
    # Compute exp(x - max_val) for both parts
    exp_in0 = tl.exp(in0_vec - max_val)
    exp_einsum = tl.exp(einsum_result - max_val)
    
    # Sum of exponentials
    sum_exp = tl.sum(exp_in0) + tl.sum(exp_einsum)
    
    # Normalize to get softmax
    softmax_in0 = exp_in0 / sum_exp
    softmax_einsum = exp_einsum / sum_exp
    
    # Store full output [B, H, W, 2*J]
    # First J elements are from in_0, next J are from einsum
    out_full_offsets_0 = (b * stride_out_full_b + h * stride_out_full_h + 
                          w * stride_out_full_w + j_range * stride_out_full_j)
    out_full_offsets_1 = (b * stride_out_full_b + h * stride_out_full_h + 
                          w * stride_out_full_w + (J + j_range) * stride_out_full_j)
    tl.store(out_full_ptr + out_full_offsets_0, softmax_in0, mask=j_mask)
    tl.store(out_full_ptr + out_full_offsets_1, softmax_einsum, mask=j_mask)
    
    # Store sliced output [B, H, W, J] (first J elements of softmax)
    out_slice_offsets = (b * stride_out_slice_b + h * stride_out_slice_h + 
                         w * stride_out_slice_w + j_range * stride_out_slice_j)
    tl.store(out_slice_ptr + out_slice_offsets, softmax_in0, mask=j_mask)

@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    # Get dimensions
    B, H, W, J = in_0.shape
    C = in_1.shape[1]  # key: [B, C, H, J]
    
    # Allocate output tensors
    out_full = torch.empty(B, H, W, 2*J, device=in_0.device, dtype=in_0.dtype)
    out_slice = torch.empty(B, H, W, J, device=in_0.device, dtype=in_0.dtype)
    
    # Grid: one program per (b, h, w) position
    grid = (B * H * W,)
    
    # Block sizes (must be >= actual sizes for proper masking)
    BLOCK_C = 64
    BLOCK_J = 64
    
    # Launch kernel
    fused_einsum_cat_softmax_kernel[grid](
        in_0, in_1, in_2, out_full, out_slice,
        B, H, W, C, J,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out_full.stride(0), out_full.stride(1), out_full.stride(2), out_full.stride(3),
        out_slice.stride(0), out_slice.stride(1), out_slice.stride(2), out_slice.stride(3),
        BLOCK_C=BLOCK_C,
        BLOCK_J=BLOCK_J,
    )
    
    return (out_full, out_slice)

# Replacement function - returns the kernel wrapper function
def replacement_func():
    return fused_einsum_cat_softmax