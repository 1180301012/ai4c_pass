import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_kernel(
    conv_out_ptr, output_ptr,
    B, H: tl.constexpr, W: tl.constexpr,
    cb, chw,
    ob, ohw,
    BLOCK_HW: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    b = tl.program_id(0)
    HW = H * W

    # Phase 1: Find max over all HW positions for this batch
    max_val = -float('inf')
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        conv_val = tl.load(conv_out_ptr + b * cb + hw_offsets * chw,
                           mask=hw_mask, other=-float('inf')).to(tl.float32)
        local_max = tl.max(tl.where(hw_mask, conv_val, -float('inf')))
        max_val = tl.maximum(max_val, local_max)

    # Phase 2: Compute exp and accumulate sum
    sum_exp = 0.0
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        conv_val = tl.load(conv_out_ptr + b * cb + hw_offsets * chw,
                           mask=hw_mask, other=-float('inf')).to(tl.float32)
        exp_val = tl.exp(conv_val - max_val)
        sum_exp += tl.sum(tl.where(hw_mask, exp_val, 0.0))

    # Phase 3: Normalize and store to output
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        conv_val = tl.load(conv_out_ptr + b * cb + hw_offsets * chw,
                           mask=hw_mask, other=-float('inf')).to(tl.float32)
        exp_val = tl.exp(conv_val - max_val)
        result = exp_val / sum_exp
        tl.store(output_ptr + b * ob + hw_offsets * ohw, result.to(OUTPUT_DTYPE), mask=hw_mask)


@triton.jit
def pointwise_conv_kernel(
    input_ptr, weight_ptr, bias_ptr, conv_out_ptr,
    B, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    ib, ic, ih, iw,
    cb, chw,
    wc,
    BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    hw_block = tl.program_id(1)
    HW = H * W

    hw_start = hw_block * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets % W

    bias_val = tl.load(bias_ptr).to(tl.float32)

    # Compute 1x1 conv: dot product over C channels for each spatial position
    conv_val = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        w_vals = tl.load(weight_ptr + c_offsets * wc, mask=c_mask, other=0.0).to(tl.float32)

        i_offsets = (b * ib +
                     c_offsets[:, None] * ic +
                     h_offsets[None, :] * ih +
                     w_offsets[None, :] * iw)
        i_vals = tl.load(input_ptr + i_offsets,
                         mask=c_mask[:, None] & hw_mask[None, :], other=0.0).to(tl.float32)

        conv_val += tl.sum(i_vals * w_vals[:, None], axis=0)

    conv_val += bias_val
    # Store only valid positions; masked positions don't matter for correctness
    # since softmax kernel reads them with mask and other=-inf
    tl.store(conv_out_ptr + b * cb + hw_offsets * chw, conv_val, mask=hw_mask)


@torch.fx.wrap
def fused_conv_view_softmax(input, weight, bias):
    B = input.shape[0]
    C = input.shape[1]  # 512
    H = input.shape[2]  # 64
    W = input.shape[3]  # 64
    HW = H * W  # 4096

    # Allocate buffers
    conv_out = torch.empty((B, HW), dtype=torch.float32, device=input.device)
    output = torch.empty((B, 1, HW), dtype=input.dtype, device=input.device)

    # Dtype mapping
    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    OUTPUT_DTYPE = dtype_map[input.dtype]

    # Get strides
    ib, ic, ih, iw = input.stride()
    ob, o1, ohw = output.stride()
    cb, chw = conv_out.stride()
    wc = weight.stride(1)

    BLOCK_HW_CONV = 256
    BLOCK_C = 32
    HW_BLOCKS = triton.cdiv(HW, BLOCK_HW_CONV)

    BLOCK_HW_SOFTMAX = 256

    # Kernel 1: Compute 1x1 conv output
    grid_conv = (B, HW_BLOCKS)
    pointwise_conv_kernel[grid_conv](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias,
        conv_out_ptr=conv_out,
        B=B, C=C, H=H, W=W,
        ib=ib, ic=ic, ih=ih, iw=iw,
        cb=cb, chw=chw,
        wc=wc,
        BLOCK_HW=BLOCK_HW_CONV, BLOCK_C=BLOCK_C,
    )

    # Kernel 2: Fused softmax (one program per batch)
    grid_softmax = (B,)
    fused_softmax_kernel[grid_softmax](
        conv_out_ptr=conv_out, output_ptr=output,
        B=B, H=H, W=W,
        cb=cb, chw=chw,
        ob=ob, ohw=ohw,
        BLOCK_HW=BLOCK_HW_SOFTMAX,
        OUTPUT_DTYPE=OUTPUT_DTYPE,
    )

    return output


def replacement_func():
    return fused_conv_view_softmax