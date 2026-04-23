import torch
import triton
import triton.language as tl


# ---- Triton kernel for batch_norm (inference mode) ----

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_C': 128}, num_warps=2),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 384}, num_warps=4),
        triton.Config({'BLOCK_C': 512}, num_warps=8),
    ],
    key=['C'],
)
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

    # Load parameters
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
def fused_dispatch_wrapper(*args):
    route = args[-1]
    args = args[:-1]

    if route == "bn":
        input_tensor, running_mean, running_var, bn_weight, bn_bias = args
        B = input_tensor.shape[0]
        C = input_tensor.shape[1]

        output = torch.empty((B, C), dtype=input_tensor.dtype, device=input_tensor.device)

        batch_norm_inference_kernel[(B,)](
            input_tensor, running_mean, running_var, bn_weight, bn_bias, output,
            B, C,
        )
        return output

    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return fused_dispatch_wrapper