import torch
import triton
import triton.language as tl

def pattern(in_1, in_3, in_2):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 16, 31)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 17, 31)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))]
    tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
    tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    return tmp_10

def replacement_args(in_1, in_3, in_2):
    return (in_1, in_2, in_3)

@triton.jit
def optimized_kernel(
    in_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Map flattened index to 5D tensor: [4,16,16,16,16]
    b = (offsets // (16*16*16*16)) % 4
    h = (offsets // (16*16*16)) % 16
    h2 = (offsets // (16*16)) % 16
    w = (offsets // 16) % 16
    w2 = offsets % 16

    # Input X has shape [4,16,16,31], so compute index for the value
    x_w = 15 + w2  # Width in input X
    x_idx = b * (16*16*31) + h * (16*31) + h2 * 31 + x_w
    value = tl.load(in_ptr + x_idx, mask=(x_w < 31), other=0.0)

    tl.store(out_ptr + offsets, value, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_1, in_2, in_3):
    # Compute the matrix multiply
    X = torch.matmul(in_1, in_3)
    # Output tensor shape: [4,16,16,16,16]
    out = torch.empty([4,16,16,16,16], dtype=X.dtype, device=X.device)
    n_elements = 4*16*16*16*16
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    optimized_kernel[(num_programs,)](
        in_ptr=X,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return kernel_wrapper