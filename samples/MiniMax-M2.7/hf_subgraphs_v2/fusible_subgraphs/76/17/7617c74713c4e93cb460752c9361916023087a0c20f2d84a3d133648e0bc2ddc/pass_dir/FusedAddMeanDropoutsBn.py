import torch
import triton
import triton.language as tl


def pattern(in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def fused_add_mean_kernel(
    in_4_ptr, in_5_ptr, out_ptr,
    in_4_stride_0, in_4_stride_1, in_4_stride_2, in_4_stride_3,
    in_5_stride_0, in_5_stride_1, in_5_stride_2, in_5_stride_3,
    out_stride_0, out_stride_1,
    N, C, H, W,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (n, c)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Offsets
    c_offsets = pid_c + tl.arange(0, 1)
    c_mask = c_offsets < C
    
    # Use tl.zeros for accumulator to ensure consistent type
    sum_vals = tl.zeros((1,), dtype=tl.float32)
    
    for hw in range(HW):
        h = hw // W
        w = hw % W
        
        # in_4 offset: n*stride_0 + c*stride_1 + h*stride_2 + w*stride_3
        in_4_offset = pid_n * in_4_stride_0 + c_offsets * in_4_stride_1 + h * in_4_stride_2 + w * in_4_stride_3
        in_5_offset = pid_n * in_5_stride_0 + c_offsets * in_5_stride_1 + h * in_5_stride_2 + w * in_5_stride_3
        
        val_4 = tl.load(in_4_ptr + in_4_offset, mask=c_mask, other=0.0)
        val_5 = tl.load(in_5_ptr + in_5_offset, mask=c_mask, other=0.0)
        
        sum_vals = sum_vals + val_4 + val_5
    
    mean_val = sum_vals / tl.cast(HW, tl.float32)
    
    out_offset = pid_n * out_stride_0 + c_offsets * out_stride_1
    tl.store(out_ptr + out_offset, mean_val, mask=c_mask)


def fused_add_mean(in_4, in_5):
    N, C, H, W = in_4.shape
    HW = H * W
    
    out = torch.empty((N, C), dtype=torch.float32, device=in_4.device)
    
    in_4_stride_0, in_4_stride_1, in_4_stride_2, in_4_stride_3 = in_4.stride()
    in_5_stride_0, in_5_stride_1, in_5_stride_2, in_5_stride_3 = in_5.stride()
    out_stride_0, out_stride_1 = out.stride()
    
    # 2D grid for parallelism over N and C
    grid = (N, C)
    
    fused_add_mean_kernel[grid](
        in_4, in_5, out,
        in_4_stride_0, in_4_stride_1, in_4_stride_2, in_4_stride_3,
        in_5_stride_0, in_5_stride_1, in_5_stride_2, in_5_stride_3,
        out_stride_0, out_stride_1,
        N, C, H, W,
        HW,
    )
    
    return out.to(in_4.dtype)


def replacement_func():
    return fused_add_mean