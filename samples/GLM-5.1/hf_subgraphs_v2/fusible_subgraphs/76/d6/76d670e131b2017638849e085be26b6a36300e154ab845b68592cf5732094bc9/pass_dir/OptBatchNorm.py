import torch
import triton
import triton.language as tl


@triton.jit
def batch_norm_inference_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= B:
        return

    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C

    running_mean = tl.load(running_mean_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    running_var = tl.load(running_var_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)

    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale = weight * inv_std
    shift = bias - running_mean * scale

    input_row = tl.load(input_ptr + row_idx * C + c_offsets, mask=mask, other=0.0).to(tl.float32)
    output_row = input_row * scale + shift
    tl.store(output_ptr + row_idx * C + c_offsets, output_row, mask=mask)


@torch.fx.wrap
def triton_batch_norm_inference(input_tensor, running_mean, running_var, bn_weight, bn_bias):
    B = input_tensor.shape[0]
    C = input_tensor.shape[1]
    output = torch.empty((B, C), dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_C = triton.next_power_of_2(C)
    batch_norm_inference_kernel[(B,)](
        input_tensor, running_mean, running_var, bn_weight, bn_bias, output,
        B, C,
        BLOCK_C=BLOCK_C,
    )
    return output


def pattern(in_7, in_0, in_1, in_3, in_2):
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)


def replacement_func():
    return triton_batch_norm_inference