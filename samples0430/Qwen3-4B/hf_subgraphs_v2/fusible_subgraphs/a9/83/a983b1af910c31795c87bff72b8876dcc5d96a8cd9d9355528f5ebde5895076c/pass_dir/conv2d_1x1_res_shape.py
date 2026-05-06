import torch
import triton
import triton.language as tl

def pattern(in_2: torch.Tensor, in_1: torch.Tensor, in_0: torch.Tensor) -> torch.Tensor:
    """Matches the 1x1 convolution followed by scaling (1.0) and reshape pattern."""
    conv_out = torch.conv2d(
        in_2,
        in_1,
        in_0,
        (1, 1),
        (0, 0),
        (1, 1),
        1
    )
    scaled = conv_out * 1.0
    return scaled.reshape(-1, 17, 4096)

def replacement_args(in_2: torch.Tensor, in_1: torch.Tensor, in_0: torch.Tensor) -> tuple:
    """Extracts arguments needed for the kernel."""
    return (in_2, in_1, in_0)

@triton.jit
def optimized_kernel(
    in_2_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    block_size: tl.constexpr,
):
    # Process each block
    pid = tl.program_id(0)
    out_val = tl.load(in_2_ptr + pid) * tl.load(in_1_ptr + pid) + tl.load(in_0_ptr + pid)
    tl.store(out_ptr + pid, out_val)
@torch.fx.wrap
def kernel_wrapper(
    in_2: torch.Tensor,
    in_1: torch.Tensor,
    in_0: torch.Tensor
):
    grid = (1024,)
    out = torch.empty_like(in_2)
    optimized_kernel[grid](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        batch_size=in_2.shape[0],
        in_channels=in_2.shape[1],
        out_channels=in_1.shape[0],
        H=in_2.shape[2],
        W=in_2.shape[3],
        block_size=1024,
    )
    return out
def replacement_func():
    return kernel_wrapper