import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    linear_out = torch.nn.functional.linear(in_2, in_1, in_0)
    permuted = linear_out.permute(0, 2, 1)
    reshaped = permuted.reshape(2, -1, 16, 16)
    interpolated = torch.nn.functional.interpolate(reshaped, size=(128, 128), mode='bilinear', align_corners=False)
    return interpolated
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def interpolate_kernel(
    input_ptr: tl.pointer,
    output_ptr: tl.pointer,
    N: tl.int32,
    C: tl.int32,
    H_in: tl.int32,
    W_in: tl.int32,
    H_out: tl.int32,
    W_out: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program block handles a portion of the output
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Placeholder: simply copy the input to output
    for i in range(BLOCK_SIZE):
        tl.store(output_ptr + block_start + i, tl.load(input_ptr + i))

def kernel_wrapper(in_0, in_1, in_2):
    N = in_0.shape[0]
    C = in_0.shape[1]
    H_in = 16
    W_in = 16
    H_out = 128
    W_out = 128
    output = torch.empty((N, C, H_out, W_out), device=in_0.device, dtype=in_0.dtype)
    # Grid size: number of blocks in height and width
    grid = (H_out // 16, W_out // 16)
    interpolate_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        N=N,
        C=C,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        BLOCK_SIZE=128,
    )
    return output
def replacement_func():
    return kernel_wrapper