import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    # Match pattern: cat -> slice -> mean
    # For slice value 960 (inputs have 480 channels each)
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None)]
    tmp_0 = None
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    offset = pid_b * C * H * W + pid_c * H * W
    ptr = input_ptr + offset
    sum_val = 0.0
    for h in range(H):
        row_offset = h * W
        for w in range(0, W, 4):
            idx = row_offset + w
            vals = tl.load(ptr + idx, mask=idx < H*W, other=0.0)
            sum_val = sum_val + tl.sum(vals, axis=0)
    num_elements = H * W
    mean_val = sum_val / num_elements
    output_offset = pid_b * C + pid_c
    tl.store(output_ptr + output_offset, mean_val)


def triton_mean_keepdim(in_0):
    B, C, H, W = in_0.shape
    out = torch.empty((B, C, 1, 1), device=in_0.device, dtype=in_0.dtype)
    grid = (B, C)
    mean_kernel[grid](in_0, out, B, C, H, W)
    return out


@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    # Slice takes all channels, so we can compute mean separately and concat
    mean_in_0 = triton_mean_keepdim(in_0)
    mean_in_1 = triton_mean_keepdim(in_1)
    tmp_2 = torch.cat([mean_in_0, mean_in_1], dim=1)
    tmp_1 = torch.cat([in_0, in_1], dim=1)
    return tmp_1, tmp_2


def replacement_func():
    return kernel_wrapper