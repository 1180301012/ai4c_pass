import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Simple pattern: just mul and sum"""
    tmp_4 = in_1 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return (tmp_6,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def simple_mul_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    batch_size, num_splits, num_channels, height, width, hw_total,
    stride_in0_b, stride_in0_s, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_s, stride_in1_c, stride_in1_h, stride_in1_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    b_idx = pid_b
    c_idx = pid_c
    
    hw_start = pid_hw * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask = hw_offsets < hw_total
    
    w_indices = hw_offsets % width
    h_indices = hw_offsets // width
    
    # Sum over num_splits dimension
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for s in range(num_splits):
        offset_in0 = (b_idx * stride_in0_b + s * stride_in0_s + 
                     c_idx * stride_in0_c + h_indices * stride_in0_h + w_indices * stride_in0_w)
        offset_in1 = (b_idx * stride_in1_b + s * stride_in1_s + 
                     c_idx * stride_in1_c + h_indices * stride_in1_h + w_indices * stride_in1_w)
        
        val_in0 = tl.load(in_0_ptr + offset_in0, mask=hw_mask, other=0.0)
        val_in1 = tl.load(in_1_ptr + offset_in1, mask=hw_mask, other=0.0)
        
        result += val_in0 * val_in1
    
    offset_out = b_idx * stride_out_b + c_idx * stride_out_c + h_indices * stride_out_h + w_indices * stride_out_w
    tl.store(out_ptr + offset_out, result, mask=hw_mask)


@torch.fx.wrap
def simple_mul_sum(in_0, in_1):
    batch_size, num_splits, num_channels, height, width = in_0.shape
    hw_total = height * width
    
    out = torch.empty((batch_size, num_channels, height, width), 
                      device=in_0.device, dtype=in_0.dtype)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch_size, num_channels, triton.cdiv(hw_total, BLOCK_SIZE))
    
    simple_mul_sum_kernel[grid](
        in_0, in_1, out,
        batch_size, num_splits, num_channels, height, width, hw_total,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3), in_0.stride(4),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3), in_1.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=1024,
    )
    
    return out


def replacement_func():
    return simple_mul_sum