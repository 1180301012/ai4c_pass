import torch
import triton
import triton.language as tl

# Pattern matching function for batch size 32
def pattern(in_0, in_1, in_2):
    # in_0: bias [1]
    # in_1: weight [1, 512, 1, 1]
    # in_2: input [32, 512, 64, 64]
    conv_out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    viewed = conv_out.view(32, 1, -1)
    softmax_out = viewed.softmax(dim=-1)
    return softmax_out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def conv1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C,
    HW,
    stride_b,
    stride_c,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW
    
    bias = tl.load(bias_ptr)
    acc = tl.full([BLOCK_HW], bias, dtype=tl.float32)
    
    batch_offset = pid_b * stride_b
    
    for c in range(C):
        w = tl.load(weight_ptr + c)
        input_offset = batch_offset + c * stride_c + hw_offsets
        inp = tl.load(input_ptr + input_offset, mask=hw_mask, other=0.0)
        acc += inp * w
    
    output_offset = pid_b * HW + hw_offsets
    tl.store(output_ptr + output_offset, acc, mask=hw_mask)


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * HW
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))
    
    max_val = tl.max(x, axis=0)
    x_exp = tl.exp(x - max_val)
    sum_exp = tl.sum(tl.where(mask, x_exp, 0.0), axis=0)
    result = x_exp / sum_exp
    
    tl.store(output_ptr + row_start + offsets, result, mask=mask)


@torch.fx.wrap
def fused_conv1x1_view_softmax(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    HW = H * W
    
    weight_flat = in_1.view(-1)
    stride_b = C * HW
    stride_c = HW
    
    conv_out = torch.empty((B, HW), dtype=in_2.dtype, device=in_2.device)
    
    # Use larger block for better performance
    BLOCK_HW = 512
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    
    conv1x1_kernel[(B, num_hw_blocks)](
        in_2,
        weight_flat,
        in_0,
        conv_out,
        C=C,
        HW=HW,
        stride_b=stride_b,
        stride_c=stride_c,
    )
    
    result = torch.empty((B, 1, HW), dtype=in_2.dtype, device=in_2.device)
    
    softmax_kernel[(B,)](
        conv_out,
        result.view(B, -1),
        HW=HW,
        BLOCK_SIZE=4096,
    )
    
    return result


def replacement_func():
    return fused_conv1x1_view_softmax