import torch
import triton
import triton.language as tl


# Match only the post-conv epilogue. Leaving conv2d untouched lets cuDNN keep
# the dominant compute path, while we fuse the three memory-bound epilogue ops.
def pattern(x, running_mean, running_var, bn_bias, bn_weight, residual):
    y = torch.nn.functional.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    y = torch.nn.functional.leaky_relu(y, 0.01, True)
    y = y + residual
    return y


# Keep replacement_args as a pure passthrough so the rewritten FX graph does not
# introduce extra eager pointwise ops each invocation.
def replacement_args(x, running_mean, running_var, bn_bias, bn_weight, residual):
    return (x, residual, running_mean, running_var, bn_bias, bn_weight)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements", "HW"],
)
@triton.jit
def _bn_lrelu_add_1d_kernel(
    x_ptr,
    residual_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0).to(tl.float32)

    ch = (offsets // HW) % C
    mean = tl.load(mean_ptr + ch, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + ch, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + ch, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + ch, mask=mask, other=0).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + 1e-05) * weight + bias
    y = tl.where(y >= 0, y, y * 0.01)
    out = y + residual

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 2, "BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 256}, num_warps=8),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_warps=8),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=8),
    ],
    key=["C", "HW"],
)
@triton.jit
def _bn_lrelu_add_2d_kernel(
    x_ptr,
    residual_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    c_mask = c_offsets < C
    hw_mask = hw_offsets < HW
    mask = c_mask[:, None] & hw_mask[None, :]

    base = (pid_n * C + c_offsets[:, None]) * HW + hw_offsets[None, :]

    x = tl.load(x_ptr + base, mask=mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask, other=0).to(tl.float32)

    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0).to(tl.float32)[:, None]
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1).to(tl.float32)[:, None]
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0).to(tl.float32)[:, None]
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0).to(tl.float32)[:, None]

    y = (x - mean) * tl.rsqrt(var + 1e-05) * weight + bias
    y = tl.where(y >= 0, y, y * 0.01)
    out = y + residual

    tl.store(out_ptr + base, out, mask=mask)


@torch.fx.wrap
def fused_bn_lrelu_add(x, residual, running_mean, running_var, bn_bias, bn_weight):
    n = x.shape[0]
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    n_elements = x.numel()

    # Overwrite the conv output in-place. In the matched graph this tensor is a
    # fresh temporary consumed only by the BN epilogue, so reusing its storage
    # removes an allocation and reduces memory traffic.
    out = x

    # Small single-image activations are launch-overhead sensitive; a flat kernel
    # keeps the grid compact. Large tensors benefit from explicit channel/HW tiling.
    if n_elements <= 1_048_576:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _bn_lrelu_add_1d_kernel[grid](
            x,
            residual,
            running_mean,
            running_var,
            bn_bias,
            bn_weight,
            out,
            n_elements,
            c,
            hw,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(hw, meta["BLOCK_HW"]),
            triton.cdiv(c, meta["BLOCK_C"]),
            n,
        )
        _bn_lrelu_add_2d_kernel[grid](
            x,
            residual,
            running_mean,
            running_var,
            bn_bias,
            bn_weight,
            out,
            c,
            hw,
        )
    return out


def replacement_func():
    return fused_bn_lrelu_add