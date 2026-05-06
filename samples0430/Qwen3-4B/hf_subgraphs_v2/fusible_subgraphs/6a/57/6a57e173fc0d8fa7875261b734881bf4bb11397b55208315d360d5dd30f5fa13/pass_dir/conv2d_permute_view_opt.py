import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv = torch.conv2d(
        in_2,
        in_0,
        None,
        (1, 1),
        (32, 0),
        (1, 1),
        4
    )
    result = in_1 + conv
    permuted = result.permute(0, 2, 1, 3)
    contiguous_tensor = permuted.contiguous()
    output = contiguous_tensor.view(128, 64, 32)
    return output

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    B: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the thread index
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offset)
    in_1 = tl.load(in_1_ptr + offset)
    in_2 = tl.load(in_2_ptr + offset)
    
    # Simulate the operations (for demonstration)
    conv = in_2 + in_0
    result = in_1 + conv
    permuted = result.permute(0, 2, 1, 3)
    contiguous_tensor = permuted.contiguous()
    
    # Convert to output shape (128, 64, 32)
    out = contiguous_tensor.view(128, 64, 32)
    
    # Store output
    tl.store(out_ptr + offset, out)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    B = in_1.shape[0]
    H = in_1.shape[2]
    W = in_1.shape[3]
    out = torch.empty((B, 128, 64, 32),
                      device=in_0.device,
                      dtype=in_0.dtype)
    
    optimized_kernel[(tl.cdiv(B, 128),)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        B=B,
        H=H,
        W=W,
        BLOCK_SIZE=128,
    )
    return out

def replacement_func():
    return kernel_wrapper