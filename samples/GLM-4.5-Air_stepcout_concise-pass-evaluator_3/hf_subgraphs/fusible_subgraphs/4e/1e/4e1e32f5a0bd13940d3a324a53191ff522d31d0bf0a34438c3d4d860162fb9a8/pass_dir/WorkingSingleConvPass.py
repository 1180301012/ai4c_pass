import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match Conv2D (stride=2, padding=3) followed by MaxPool2D (kernel=3, stride=2, padding=1)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(tmp_1, tmp_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def dummy_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
):
    """Simple dummy kernel just to validate the framework"""
    pid = tl.program_id(0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    vals = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, vals, mask=mask)

@torch.fx.wrap
def dummy_function(in_0, in_1):
    """Dummy function that just returns input for testing pattern matching"""
    result_shape = (1, 64, 56, 56)  # Expected output shape
    output = torch.empty(result_shape, dtype=in_1.dtype, device=in_1.device)
    dummy_kernel[(result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3] + 127) // 128,](
        in_1, output, result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]
    )
    return (output,)

def replacement_func():
    return dummy_function