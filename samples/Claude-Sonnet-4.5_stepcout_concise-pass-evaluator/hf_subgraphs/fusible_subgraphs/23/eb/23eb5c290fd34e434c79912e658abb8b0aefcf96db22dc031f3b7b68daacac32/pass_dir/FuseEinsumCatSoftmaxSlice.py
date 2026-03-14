import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Pattern matching for einsum + cat + softmax + slice fusion
    """
    tmp_0 = in_0
    tmp_1 = torch.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_J': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_J': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_J': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_J': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_J': 32}, num_warps=8),
    ],
    key=['C', 'H', 'W', 'J'],
)
@triton.jit
def fused_einsum_cat_softmax_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_full_ptr, out_slice_ptr,
    B, C, H, W, J,
    stride_in0_b, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_j,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_b, stride_out_h, stride_out_w, stride_out_j,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_J: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Compute einsum: 'bchw,bchj->bhwj'
    # For each output position [b, h, w, j], we sum over c: in_2[b,c,h,w] * in_1[b,c,h,j]
    
    # Allocate space for concatenated result (size 128 = 64 + 64)
    concat_result = tl.zeros([128], dtype=tl.float32)
    
    # First part: copy in_0[b, :, h, w] to concat_result[:64]
    # in_0 has shape [B, C, H, W], we want [b, :, h, w]
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    in0_base = pid_b * stride_in0_b + pid_h * stride_in0_h + pid_w * stride_in0_w
    in0_offsets = in0_base + c_offsets * stride_in0_c
    in0_vals = tl.load(in_0_ptr + in0_offsets, mask=c_mask, other=0.0)
    
    # Store first 64 values to concat_result
    for i in range(64):
        if i < C:
            concat_result[i] = in0_vals[i]
    
    # Second part: compute einsum for each j and store to concat_result[64:128]
    for j_idx in range(J):
        # Compute sum over c: in_2[b,c,h,w] * in_1[b,c,h,j]
        acc = tl.zeros([1], dtype=tl.float32)
        
        for c_block_start in range(0, C, BLOCK_SIZE_C):
            c_offsets = c_block_start + tl.arange(0, BLOCK_SIZE_C)
            c_mask = c_offsets < C
            
            # Load in_2[b, c, h, w]
            in2_base = pid_b * stride_in2_b + pid_h * stride_in2_h + pid_w * stride_in2_w
            in2_offsets = in2_base + c_offsets * stride_in2_c
            in2_vals = tl.load(in_2_ptr + in2_offsets, mask=c_mask, other=0.0)
            
            # Load in_1[b, c, h, j]
            in1_base = pid_b * stride_in1_b + pid_h * stride_in1_h + j_idx * stride_in1_j
            in1_offsets = in1_base + c_offsets * stride_in1_c
            in1_vals = tl.load(in_1_ptr + in1_offsets, mask=c_mask, other=0.0)
            
            # Multiply and accumulate
            acc += tl.sum(in2_vals * in1_vals)
        
        # Store to concat_result[64 + j_idx]
        concat_result[64 + j_idx] = acc
    
    # Apply softmax on concat_result
    # Step 1: Find max
    max_val = concat_result[0]
    for i in range(1, 128):
        if concat_result[i] > max_val:
            max_val = concat_result[i]
    
    # Step 2: Compute exp and sum
    exp_sum = 0.0
    for i in range(128):
        concat_result[i] = tl.exp(concat_result[i] - max_val)
        exp_sum += concat_result[i]
    
    # Step 3: Normalize
    for i in range(128):
        concat_result[i] = concat_result[i] / exp_sum
    
    # Write outputs
    out_base = pid_b * stride_out_b + pid_h * stride_out_h + pid_w * stride_out_w
    
    # Write full softmax output
    for j_idx in range(128):
        out_offset = out_base + j_idx * stride_out_j
        tl.store(out_full_ptr + out_offset, concat_result[j_idx])
    
    # Write sliced output (first 64 elements)
    for j_idx in range(64):
        out_offset = out_base + j_idx * stride_out_j
        tl.store(out_slice_ptr + out_offset, concat_result[j_idx])


@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    J = in_1.shape[3]
    
    # Output shapes
    out_full = torch.empty((B, H, W, 128), device=in_0.device, dtype=in_0.dtype)
    out_slice = torch.empty((B, H, W, 64), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel
    grid = (B, H, W)
    
    fused_einsum_cat_softmax_kernel[grid](
        in_0, in_1, in_2,
        out_full, out_slice,
        B, C, H, W, J,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out_full.stride(0), out_full.stride(1), out_full.stride(2), out_full.stride(3),
    )
    
    return out_full, out_slice


def replacement_func():
    return fused_einsum_cat_softmax