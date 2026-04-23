import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_reshape_avgpool_bn_silu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
    output_ptr,
    n_elements_output,
    input_stride_1,
    input_stride_2,
    C: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_output

    # Output layout: [1, C, H_out, W_out]
    # For each output element, compute which channel and spatial position
    c = offsets // (H_out * W_out)
    spatial = offsets % (H_out * W_out)
    h_out = spatial // W_out
    w_out = spatial % W_out

    # For avg_pool2d with kernel_size=2, stride=2:
    # h_in = h_out * 2, w_in = w_out * 2
    # Average of 4 values: (h_in, w_in), (h_in, w_in+1), (h_in+1, w_in), (h_in+1, w_in+1)
    h_in_base = h_out * 2
    w_in_base = w_out * 2

    # Input layout after reshape: [1, C, 16, 16] -> contiguous strides
    # stride_1 = 256 (16*16), stride_2 = 16, stride_3 = 1

    avg_val0 = tl.load(input_ptr + c * input_stride_1 + h_in_base * input_stride_2 + w_in_base,
                       mask=mask, other=0.0)
    avg_val1 = tl.load(input_ptr + c * input_stride_1 + h_in_base * input_stride_2 + (w_in_base + 1),
                       mask=mask, other=0.0)
    avg_val2 = tl.load(input_ptr + c * input_stride_1 + (h_in_base + 1) * input_stride_2 + w_in_base,
                       mask=mask, other=0.0)
    avg_val3 = tl.load(input_ptr + c * input_stride_1 + (h_in_base + 1) * input_stride_2 + (w_in_base + 1),
                       mask=mask, other=0.0)

    avg_pool_result = (avg_val0 + avg_val1 + avg_val2 + avg_val3) * 0.25

    # BatchNorm: output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
    mean_val = tl.load(running_mean_ptr + c, mask=mask, other=0.0)
    var_val = tl.load(running_var_ptr + c, mask=mask, other=1.0)
    weight_val = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)

    bn_result = (avg_pool_result - mean_val) / tl.sqrt(var_val + 1e-05) * weight_val + bias_val

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu_result = bn_result * tl.sigmoid(bn_result)

    tl.store(output_ptr + offsets, silu_result, mask=mask)


@torch.fx.wrap
def fused_reshape_avgpool_bn_silu(in_0, in_1, in_2, in_3, in_4):
    # in_4 shape: [4, 128, 256] -> reshape to [1, 512, 16, 16]
    C = 512
    H_in = 16
    W_in = 16
    H_out = 8
    W_out = 8
    n_elements_output = C * H_out * W_out  # 512 * 8 * 8 = 32768

    # Create output tensor
    output = torch.empty((1, C, H_out, W_out), dtype=in_4.dtype, device=in_4.device)

    # Strides for reshaped view [1, 512, 16, 16] of contiguous input [4, 128, 256]
    input_stride_1 = H_in * W_in  # 256
    input_stride_2 = W_in          # 16

    BLOCK_SIZE = 512
    num_programs = (n_elements_output + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_reshape_avgpool_bn_silu_kernel[(num_programs,)](
        input_ptr=in_4,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        bias_ptr=in_2,
        weight_ptr=in_3,
        output_ptr=output,
        n_elements_output=n_elements_output,
        input_stride_1=input_stride_1,
        input_stride_2=input_stride_2,
        C=C,
        H_out=H_out,
        W_out=W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return fused_reshape_avgpool_bn_silu