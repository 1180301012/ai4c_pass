import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[(None, None, slice(None, None, None))]
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
    ],
    key=['num_rows'],
)
@triton.jit
def silu_split_kernel(
    in_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < 1152

    # Load input row
    x = tl.load(in_ptr + row_idx * 1152 + cols, mask=mask)

    # SiLU in float32 for numerical accuracy
    x_f32 = x.to(tl.float32)
    silu_x = (x_f32 * tl.sigmoid(x_f32)).to(x.dtype)

    # Store to out0: first 512 elements
    mask0 = cols < 512
    tl.store(out0_ptr + row_idx * 512 + cols, silu_x, mask=mask0)

    # Store to out1: elements 512-1023
    mask1 = (cols >= 512) & (cols < 1024)
    tl.store(out1_ptr + row_idx * 512 + (cols - 512), silu_x, mask=mask1)

    # Store to out2: elements 1024-1151
    mask2 = (cols >= 1024) & (cols < 1152)
    tl.store(out2_ptr + row_idx * 128 + (cols - 1024), silu_x, mask=mask2)


@triton.jit
def copy_reshape_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_silu_split_reshape(in_0, in_1):
    B = in_1.shape[0]
    S = in_1.shape[1]
    num_rows = B * S

    # Allocate contiguous output tensors
    out0 = torch.empty((B, S, 512), dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty((B, S, 512), dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((B, S, 1, 128), dtype=in_1.dtype, device=in_1.device)

    # Launch fused SiLU + split kernel
    silu_split_kernel[(num_rows,)](
        in_1, out0, out1, out2,
        num_rows,
    )

    # Handle in_0 reshape: [2, 128] -> [1, 1, 2, 128]
    n_elements = in_0.numel()
    d0 = in_0.shape[0]
    d1 = in_0.shape[1]
    out_reshaped = torch.empty((1, 1, d0, d1), dtype=in_0.dtype, device=in_0.device)
    copy_reshape_kernel[(1,)](
        in_0, out_reshaped, n_elements, BLOCK_SIZE=256,
    )

    return (out_reshaped, out0, out2, out1)


def replacement_func():
    return fused_silu_split_reshape