import torch
import triton
import triton.language as tl


def pattern(in_5, in_1, in_2, in_4, in_3):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8


def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)


@triton.jit
def fused_pool_bn_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C

    # Load BN parameters
    mean_val = tl.load(running_mean_ptr + c).to(tl.float32)
    var_val = tl.load(running_var_ptr + c).to(tl.float32)
    w_val = tl.load(weight_ptr + c).to(tl.float32)
    b_val = tl.load(bias_ptr + c).to(tl.float32)

    # Compute global average pooling: load all HW elements and compute mean
    base_offset = pid * HW
    offsets = tl.arange(0, HW)
    x = tl.load(input_ptr + base_offset + offsets).to(tl.float32)
    avg = tl.sum(x, axis=0) / HW

    # Batch norm (eval mode): weight * (x - mean) / sqrt(var + eps) + bias
    inv_std = 1.0 / tl.sqrt(var_val + 1e-05)
    result = w_val * (avg - mean_val) * inv_std + b_val

    # ReLU
    result = tl.maximum(result, 0.0)

    # Store result
    tl.store(output_ptr + pid, result)


@torch.fx.wrap
def fused_pool_bn_relu(in_5, in_1, in_2, in_4, in_3):
    B = in_5.shape[0]
    C = in_5.shape[1]
    H = in_5.shape[2]
    W = in_5.shape[3]
    HW = H * W

    output = torch.empty((B, C, 1, 1), dtype=in_5.dtype, device=in_5.device)

    grid = (B * C,)
    fused_pool_bn_relu_kernel[grid](
        in_5, in_1, in_2, in_4, in_3, output,
        C=C,
        HW=HW,
        num_warps=2,
    )

    return output


def replacement_func():
    return fused_pool_bn_relu