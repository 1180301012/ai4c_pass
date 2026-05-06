import torch
import triton
import triton.language as tl

def pattern(in1, in0):
    return torch.conv2d(in1, in0, None, (1, 1), (0, 0), (1, 1), 1)

def replacement_args(in1, in0):
    return (in1, in0)

@triton.jit
def conv1x1_kernel(
    in1_ptr,
    in0_ptr,
    out_ptr,
    num_out_channels,
    num_spatial,
    num_in_channels,
    BLOCK_SIZE: tl.constexpr,
):
    out_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    if out_idx >= num_out_channels or spatial_idx >= num_spatial:
        return
    total = 0.0
    for i in range(num_in_channels):
        in1_val = tl.load(in1_ptr + (i * num_spatial + spatial_idx))
        in0_val = tl.load(in0_ptr + (out_idx * num_in_channels + i))
        total += in1_val * in0_val
    tl.store(out_ptr + (out_idx * num_spatial + spatial_idx), total)

@torch.fx.wrap
def kernel_wrapper(in1, in0):
    num_out_channels = in0.shape[0]
    num_in_channels = in0.shape[1]
    num_spatial = in1.shape[3]
    out = torch.empty((1, num_out_channels, 1, num_spatial), dtype=in1.dtype, device=in1.device)
    BLOCK_SIZE = 128
    num_blocks = (num_out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_spatial = (num_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    conv1x1_kernel[(num_blocks, num_blocks_spatial)](\n        in1_ptr=in1,\n        in0_ptr=in0,\n        out_ptr=out,\n        num_out_channels=num_out_channels,\n        num_spatial=num_spatial,\n        num_in_channels=num_in_channels,\n        BLOCK_SIZE=BLOCK_SIZE,\n    )
    return out

def replacement_func():
    return kernel_wrapper