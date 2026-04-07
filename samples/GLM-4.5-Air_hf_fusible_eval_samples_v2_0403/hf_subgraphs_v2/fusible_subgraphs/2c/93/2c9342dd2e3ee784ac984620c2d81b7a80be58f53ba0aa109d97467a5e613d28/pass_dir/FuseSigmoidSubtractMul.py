import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(tmp_4):
    return (tmp_4,)



@triton.jit
def fused_sigmoid_subtract_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # fused computation: sigmoid(x) - 0.25) * pi
    # Use native dtype operations where possible for better performance
    
    # For sigmoid, we need to work with float32 for numerical stability
    # but we can optimize the conversion path
    if x.dtype == tl.float32:
        # Direct computation for float32
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
        result = (sigmoid_x - 0.25) * 3.141592653589793
    elif x.dtype == tl.float16:
        # For float16, convert to float32 for sigmoid computation
        x_fp32 = x.to(tl.float32)
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x_fp32))
        result_fp32 = (sigmoid_x - 0.25) * 3.141592653589793
        # Use float32 division to get better precision from result
        result = result_fp32.to(tl.float16)
    else:  # bfloat16
        # For bfloat16, use specialized operators if available, else convert to float32
        x_fp32 = x.to(tl.float32)
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x_fp32))
        result_fp32 = (sigmoid_x - 0.25) * 3.141592653589793
        # Keep float32 for bfloat16 result if supported, else convert to bfloat16
        result = result_fp32.to(x.dtype)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_sigmoid_subtract_mul_gpu(x):
    # Determine the total number of elements
    n_elements = x.numel()
    
    # Optimized block size based on tensor size and dtype
    if x.dtype == tl.float32:
        # For float32, use larger block size for better utilization
        if n_elements < 16384:
            BLOCK_SIZE = 1024
        else:
            BLOCK_SIZE = 4096
    else:
        # For float16/bfloat16, use smaller block size for better memory bandwidth
        if n_elements < 8192:
            BLOCK_SIZE = 512
        else:
            BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Use optimized kernel configuration based on dtype
    if x.dtype == tl.float32:
        # More aggressive configuration for float32
        fused_sigmoid_subtract_mul_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,  # More warps for float32 computation
            num_stages=3,  # More stages for better pipelining
        )
    else:
        # Conservative configuration for float16/bfloat16
        fused_sigmoid_subtract_mul_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    
    return out

def replacement_func():
    return fused_sigmoid_subtract_mul_gpu