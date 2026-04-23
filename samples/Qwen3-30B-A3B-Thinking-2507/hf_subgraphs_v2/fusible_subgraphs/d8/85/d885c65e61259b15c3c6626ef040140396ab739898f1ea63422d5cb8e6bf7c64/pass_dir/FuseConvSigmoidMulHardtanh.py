import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Triton kernel
@triton.jit
def fused_conv_sigmoid_mul_hardtanh(
    in_3_ptr, in_1_ptr, in_0_ptr, in_2_ptr, output_ptr,
    batch, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // (out_channels * height * width)
    c = (pid // (height * width)) % out_channels
    i = (pid // width) % height
    j = pid % width

    # Load input channels for current batch and input
    in_3_vals = tl.load(
        in_3_ptr + b * 19 + tl.arange(0, 32),
        mask=tl.arange(0, 32) < 19,
        other=0.0
    )
    
    # Load weights for current output channel
    in_1_vals = tl.load(
        in_1_ptr + c * 19 + tl.arange(0, 32),
        mask=tl.arange(0, 32) < 19,
        other=0.0
    )

    # Dot product (convolution)
    dot = tl.sum(in_3_vals * in_1_vals, axis=0)
    conv_val = dot + tl.load(in_0_ptr + c)

    # Apply sigmoid
    conv_val_fp32 = conv_val.to(tl.float32)
    sigmoided = (1.0 / (1.0 + tl.exp(-conv_val_fp32))).to(tl.bfloat16)

    # Load input value and compute result
    in_2_val = tl.load(in_2_ptr + b * out_channels * height * width + c * height * width + i * width + j)
    mul = sigmoided * in_2_val
    result = tl.clamp(mul, 0.0, 6.0)

    tl.store(
        output_ptr + b * out_channels * height * width + c * height * width + i * width + j,
        result
    )

# Kernel wrapper
@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    batch = in_3.shape[0]
    in_channels = in_3.shape[1]
    out_channels = in_1.shape[0]
    height = in_2.shape[2]
    width = in_2.shape[3]
    output = torch.empty_like(in_2)

    num_elements = batch * out_channels * height * width
    BLOCK_SIZE = 128
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_conv_sigmoid_mul_hardtanh[(num_blocks,)](
        in_3, in_1, in_0, in_2, output,
        batch, in_channels, out_channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

# Replacement function

def replacement_func():
    return kernel_wrapper