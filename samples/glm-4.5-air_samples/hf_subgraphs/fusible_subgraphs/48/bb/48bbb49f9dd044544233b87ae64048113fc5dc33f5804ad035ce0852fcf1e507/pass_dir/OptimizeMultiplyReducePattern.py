import torch
import triton
import triton.language as tl

def pattern(reshaped_softmax, in_0, in_1):
    tmp_4 = reshaped_softmax.mul(in_0)
    tmp_5 = tmp_4.reshape(1, 17, -1)  # Base pattern, will be adapted by kernel
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    
    tmp_7 = reshaped_softmax.mul(in_1)
    tmp_8 = tmp_7.reshape(1, 17, -1)  # Base pattern, will be adapted by kernel  
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    
    return tmp_6, tmp_9

def replacement_args(reshaped_softmax, in_0, in_1):
    return (reshaped_softmax, in_0, in_1)

@triton.jit
def multiply_reduce_kernel(
    softmax_ptr,
    in0_ptr,
    in1_ptr,
    out0_ptr,
    out1_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate element-wise product and reduce
    for i in range(H * W):
        # Broadcast dimensions for element-wise multiplication
        softmax_slice = tl.load(softmax_ptr + pid * C * H * W + i, other=0.0)
        in0_val = tl.load(in0_ptr + i, other=0.0)
        in1_val = tl.load(in1_ptr + i, other=0.0)
        
        # Element-wise multiplication and reduction
        result0 = softmax_slice * in0_val
        result1 = softmax_slice * in1_val
        
        # Atomic add to output (sum reduction)
        if i == 0:
            current_val0 = tl.load(out0_ptr + pid * C + tl.arange(0, C), mask=None)
            current_val1 = tl.load(out1_ptr + pid * C + tl.arange(0, C), mask=None)
            updated_val0 = current_val0 + result0
            updated_val1 = current_val1 + result1
            tl.store(out0_ptr + pid * C + tl.arange(0, C), updated_val0)
            tl.store(out1_ptr + pid * C + tl.arange(0, C), updated_val1)

@torch.fx.wrap  
def optimized_multiply_reduce(reshaped_softmax, in_0, in_1):
    N, C, H, W = reshaped_softmax.shape
    output0 = torch.empty((N, C, 1, 1), device=reshaped_softmax.device, dtype=reshaped_softmax.dtype)
    output1 = torch.empty((N, C, 1, 1), device=reshaped_softmax.device, dtype=reshaped_softmax.dtype)
    
    # Initialize outputs to zero
    output0.fill_(0.0)
    output1.fill_(0.0)
    
    grid = lambda meta: (N * C,)
    multiply_reduce_kernel[grid](
        reshaped_softmax,
        in_0,
        in_1,
        output0,
        output1,
        N, C, H, W
    )
    
    return output0, output1

def replacement_func():
    return optimized_multiply_reduce