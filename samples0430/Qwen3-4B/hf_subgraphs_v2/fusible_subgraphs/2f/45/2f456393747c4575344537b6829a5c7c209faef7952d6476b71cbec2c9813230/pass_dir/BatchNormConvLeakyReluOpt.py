import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return tmp_8
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.jit
def optimized_kernel(
    in_0_ptr, 
    in_1_ptr, 
    in_2_ptr, 
    in_3_ptr, 
    in_4_ptr, 
    in_5_ptr, 
    in_6_ptr,
    out_ptr,
    N, 
    C_in, 
    C_out,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current block
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load inputs
    in_6 = tl.load(in_6_ptr + offsets, mask=mask, other=0.0)
    in_4 = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    in_5 = tl.load(in_5_ptr + offsets, mask=mask, other=0.0)
    
    # 1x1 convolution
    dtype = in_6.dtype
    conv = tl.zeros((BLOCK_SIZE,), dtype=dtype)
    conv = in_6 * in_4
    
    # Batch normalization
    batch_norm = (conv - in_0) / tl.sqrt(tl.cast(in_1 + 1e-5, tl.float32)) * in_3 + in_2
    
    # Leaky ReLU with slope 0.01
    leaky_relu = tl.where(batch_norm > 0, batch_norm, 0.01 * batch_norm)
    
    # Add residual connection
    out = leaky_relu + in_5
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    N = in_6.numel()
    C_in = in_4.shape[1]
    C_out = in_4.shape[0]
    
    # Creates a zero tensor of the same shape as in_6
    out = torch.empty_like(in_6)
    
    # Launch the kernel
    optimized_kernel[
        (N + 1024 - 1) // 1024,
    ](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        in_4_ptr=in_4,
        in_5_ptr=in_5,
        in_6_ptr=in_6,
        out_ptr=out,
        N=N,
        C_in=C_in,
        C_out=C_out,
        BLOCK_SIZE=1024,
    )
    
    return out
def replacement_func():
    return kernel_wrapper