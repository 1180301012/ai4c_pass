import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Exact pattern from model.py
    """
    tmp_0 = in_0
    tmp_1 = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[Ellipsis, slice(None, 64, None)]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64}, num_warps=8),
    ],
    key=['C', 'H', 'W', 'J'],
)
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_full_ptr, out_slice_ptr,
    B, C, H, W, J,
    in0_stride_b, in0_stride_c, in0_stride_h, in0_stride_w,
    in1_stride_b, in1_stride_c, in1_stride_h, in1_stride_j,
    in2_stride_b, in2_stride_c, in2_stride_h, in2_stride_w,
    out_stride_b, out_stride_h, out_stride_w, out_stride_j,
    BLOCK_C: tl.constexpr,
):
    # Get position
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_h = pid_hw // W
    pid_w = pid_hw % W
    
    # Storage for concatenated result (128 elements)
    result = tl.zeros([128], dtype=tl.float32)
    
    # First 64 elements: copy from in_0
    c_range = tl.arange(0, BLOCK_C)
    c_mask = c_range < C
    in0_offset = pid_b * in0_stride_b + pid_h * in0_stride_h + pid_w * in0_stride_w
    for c in range(0, 64, BLOCK_C):
        c_idx = c + c_range
        c_m = c_idx < 64
        val = tl.load(in_0_ptr + in0_offset + c_idx * in0_stride_c, mask=c_m & c_mask, other=0.0)
        for i in range(BLOCK_C):
            if c + i < 64 and i < BLOCK_C:
                result[c + i] = val[i]
    
    # Next 64 elements: einsum result
    for j in range(64):
        acc = 0.0
        for c_start in range(0, C, BLOCK_C):
            c_idx = c_start + c_range
            c_m = c_idx < C
            
            # Load in_2[b, c, h, w]
            in2_offset = pid_b * in2_stride_b + pid_h * in2_stride_h + pid_w * in2_stride_w
            in2_val = tl.load(in_2_ptr + in2_offset + c_idx * in2_stride_c, mask=c_m, other=0.0)
            
            # Load in_1[b, c, h, j]
            in1_offset = pid_b * in1_stride_b + pid_h * in1_stride_h + j * in1_stride_j
            in1_val = tl.load(in_1_ptr + in1_offset + c_idx * in1_stride_c, mask=c_m, other=0.0)
            
            # Accumulate
            acc += tl.sum(in2_val * in1_val)
        
        result[64 + j] = acc
    
    # Softmax
    max_val = tl.max(result)
    exp_vals = tl.exp(result - max_val)
    sum_exp = tl.sum(exp_vals)
    softmax_result = exp_vals / sum_exp
    
    # Write outputs
    out_offset = pid_b * out_stride_b + pid_h * out_stride_h + pid_w * out_stride_w
    for j in range(128):
        tl.store(out_full_ptr + out_offset + j * out_stride_j, softmax_result[j])
    for j in range(64):
        tl.store(out_slice_ptr + out_offset + j * out_stride_j, softmax_result[j])


@torch.fx.wrap
def fused_impl(in_0, in_1, in_2):
    B, C, H, W = in_0.shape
    J = in_1.shape[3]
    
    out_full = torch.empty((B, H, W, 128), device=in_0.device, dtype=in_0.dtype)
    out_slice = torch.empty((B, H, W, 64), device=in_0.device, dtype=in_0.dtype)
    
    grid = (B, H * W)
    
    fused_kernel[grid](
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
    return fused_impl