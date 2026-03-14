import torch
import triton
import triton.language as tl

# Pattern matching function for batch size 1
def pattern(in_0, in_1, in_2):
    # in_0: bias [1]
    # in_1: weight [1, 512, 1, 1]
    # in_2: input [1, 512, 64, 64]
    conv_out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    viewed = conv_out.view(1, 1, -1)
    softmax_out = viewed.softmax(dim=-1)
    return softmax_out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def conv1x1_vectorized_kernel_b1(
    input_ptr,      # [C, HW]
    weight_ptr,     # [C]
    bias_ptr,       # [1]
    output_ptr,     # [HW]
    C,
    HW,
    stride_c,       # HW
    BLOCK_HW: tl.constexpr,
):
    # 1D grid: hw_block
    pid_hw = tl.program_id(0)
    
    # Compute hw positions for this block
    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW
    
    # Initialize accumulator with bias
    bias = tl.load(bias_ptr)
    acc = tl.full([BLOCK_HW], bias, dtype=tl.float32)
    
    # Process all channels
    for c in range(C):
        # Load weight for this channel
        w = tl.load(weight_ptr + c)
        
        # Load input: input[c, hw] = input_ptr[c * stride_c + hw]
        input_offset = c * stride_c + hw_offsets
        inp = tl.load(input_ptr + input_offset, mask=hw_mask, other=0.0)
        
        # Accumulate
        acc += inp * w
    
    # Store output
    tl.store(output_ptr + hw_offsets, acc, mask=hw_mask)


@triton.jit
def softmax_row_kernel_b1(
    input_ptr,
    output_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Single program handles entire row for B=1
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Softmax: max -> exp -> sum -> normalize
    max_val = tl.max(x, axis=0)
    x_exp = tl.exp(x - max_val)
    sum_exp = tl.sum(tl.where(mask, x_exp, 0.0), axis=0)
    result = x_exp / sum_exp
    
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_conv1x1_view_softmax_b1(in_0, in_1, in_2):
    # in_0: bias [1]
    # in_1: weight [1, 512, 1, 1]
    # in_2: input [B, C, H, W]
    B, C, H, W = in_2.shape
    HW = H * W
    
    # Flatten weight
    weight_flat = in_1.view(-1)
    
    # Compute strides
    stride_c = HW
    
    # Allocate intermediate
    conv_out = torch.empty((HW,), dtype=in_2.dtype, device=in_2.device)
    
    # Launch conv kernel
    BLOCK_HW = 256
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    
    conv1x1_vectorized_kernel_b1[(num_hw_blocks,)](
        in_2,
        weight_flat,
        in_0,
        conv_out,
        C=C,
        HW=HW,
        stride_c=stride_c,
        BLOCK_HW=BLOCK_HW,
    )
    
    # Allocate output
    result = torch.empty((1, 1, HW), dtype=in_2.dtype, device=in_2.device)
    
    # Launch softmax kernel
    BLOCK_SIZE = 4096  # HW = 64*64 = 4096
    softmax_row_kernel_b1[(1,)](
        conv_out,
        result.view(-1),
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result


def replacement_func():
    return fused_conv1x1_view_softmax_b1