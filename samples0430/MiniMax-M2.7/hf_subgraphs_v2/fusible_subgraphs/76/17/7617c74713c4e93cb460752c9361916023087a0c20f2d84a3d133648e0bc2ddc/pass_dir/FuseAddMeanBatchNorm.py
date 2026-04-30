import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mean_kernel(
    x_ptr, y_ptr, 
    out_ptr,
    N, C, H, W, 
    num_warps: tl.constexpr,
    dtype: tl.constexpr
):
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc % C
    
    sum_val = tl.cast(0.0, tl.float32)
    
    # Process all spatial elements with sequential loop
    base_offset = n * C * H * W + c * H * W
    for spatial_idx in range(H * W):
        offset = base_offset + spatial_idx
        x_val = tl.load(x_ptr + offset)
        y_val = tl.load(y_ptr + offset)
        sum_val = sum_val + tl.cast(x_val, tl.float32) + tl.cast(y_val, tl.float32)
    
    mean_val = sum_val / tl.cast(H * W, tl.float32)
    
    out_offset = n * C + c
    tl.store(out_ptr + out_offset, tl.cast(mean_val, dtype))

def fused_add_mean(in_4, in_5):
    N, C, H, W = in_4.shape
    input_dtype = in_4.dtype
    
    out = torch.empty((N, C), dtype=input_dtype, device=in_4.device)
    
    grid = (N * C,)
    
    if input_dtype == torch.bfloat16:
        dtype_val = tl.bfloat16
    elif input_dtype == torch.float16:
        dtype_val = tl.float16
    else:
        dtype_val = tl.float32
    
    # Set num_warps based on H*W size
    spatial_size = H * W
    if spatial_size <= 64:
        num_warps = 2
    elif spatial_size <= 144:
        num_warps = 4
    else:
        num_warps = 8
    
    fused_add_mean_kernel[grid](
        in_4, in_5, out, N, C, H, W, num_warps, dtype_val
    )
    return out

def pattern(in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5

def replacement_args(in_4, in_5):
    return (in_4, in_5)

def replacement_func():
    return fused_add_mean