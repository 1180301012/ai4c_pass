import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(padded_tensor):
    tmp_3 = padded_tensor.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    return tmp_6

# Argument extraction function
def replacement_args(padded_tensor):
    return (padded_tensor,)

# Triton kernel
@triton.jit
def reshape_permute_kernel(
    input_ptr,
    output_ptr,
    batch, channels, height, width,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Output shape: 8,4,144,80
    i = offsets // (4*144*80)
    rem = offsets % (4*144*80)
    j = rem // (144*80)
    rem = rem % (144*80)
    k = rem // 80
    l = rem % 80

    channel = i * 80 + l
    h_patch = j // 2
    w_patch = j % 2
    patch_h = k // 12
    patch_w = k % 12
    input_h = h_patch * 8 + patch_h
    input_w = w_patch * 8 + patch_w

    input_idx = 0 * (channels * height * width) + channel * (height * width) + input_h * width + input_w
    val = tl.load(input_ptr + input_idx)
    output_idx = i * (4*144*80) + j * (144*80) + k * 80 + l
    tl.store(output_ptr + output_idx, val, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def reshape_permute_wrapper(padded_tensor):
    batch, channels, height, width = padded_tensor.shape
    num_elements = 8 * 4 * 144 * 80
    output = torch.empty(8, 4, 144, 80, device=padded_tensor.device, dtype=padded_tensor.dtype)
    BLOCK_SIZE = 1024
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    reshape_permute_kernel[(num_blocks,)](
        padded_tensor, output, batch, channels, height, width, num_elements, BLOCK_SIZE
    )

    return output

# Replacement function
def replacement_func():
    return reshape_permute_wrapper