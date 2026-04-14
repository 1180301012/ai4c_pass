import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CI': 32}),
        triton.Config({'BLOCK_CI': 64}),
        triton.Config({'BLOCK_CI': 128}),
        triton.Config({'BLOCK_CI': 256}),
        triton.Config({'BLOCK_CI': 512}),
    ],
    key=['N', 'C_in', 'C_out'],
)
@triton.jit
def _fused_kernel_38(
    x_ptr, w_ptr, b_ptr, ln_w_ptr, ln_b_ptr, out_ptr,
    N, C_in, C_out,
    eps,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.arange(0, BLOCK_CO)
    co_mask = co < C_out
    acc = tl.zeros([BLOCK_CO], dtype=tl.float32)

    for ci_start in tl.range(0, C_in, BLOCK_CI):
        ci = ci_start + tl.arange(0, BLOCK_CI)
        ci_mask = ci < C_in
        x_vals = tl.load(x_ptr + n * C_in + ci, mask=ci_mask, other=0.0).to(tl.float32)
        w_ptrs = w_ptr + co[:, None] * C_in + ci[None, :]
        w_vals = tl.load(w_ptrs, mask=co_mask[:, None] & ci_mask[None, :], other=0.0).to(tl.float32)
        acc = acc + tl.sum(w_vals * x_vals[None, :], axis=1)

    b_vals = tl.load(b_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + b_vals

    acc_sum = tl.sum(tl.where(co_mask, acc, 0.0))
    mean = acc_sum / C_out
    diff = tl.where(co_mask, acc - mean, 0.0)
    var = tl.sum(diff * diff) / C_out
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (acc - mean) * inv_std

    ln_w_vals = tl.load(ln_w_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    ln_b_vals = tl.load(ln_b_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    output = tl.where(co_mask, normalized * ln_w_vals + ln_b_vals, 0.0)
    output = tl.maximum(output, 0.0)
    tl.store(out_ptr + n * C_out + co, output.to(out_ptr.dtype.element_ty), mask=co_mask)


@torch.fx.wrap
def _triton_fused_conv1x1_ln_relu_38(in_0, in_1, in_2, in_3, in_4):
    N = in_4.shape[0]
    C_out = in_1.shape[0]
    C_in = in_1.shape[1]
    out = torch.empty((N, C_out, 1, 1), dtype=in_4.dtype, device=in_4.device)
    BLOCK_CO = triton.next_power_of_2(C_out)
    x_flat = in_4.contiguous().view(N, C_in)
    w_flat = in_1.contiguous().view(C_out, C_in)
    ln_w_flat = in_3.contiguous().view(C_out)
    ln_b_flat = in_2.contiguous().view(C_out)
    out_flat = out.view(N, C_out)
    _fused_kernel_38[(N,)](
        x_flat, w_flat, in_0, ln_w_flat, ln_b_flat, out_flat,
        N, C_in, C_out, 1e-5, BLOCK_CO=BLOCK_CO,
    )
    return out


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(conv2d, (38, 1, 1), in_3, in_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return _triton_fused_conv1x1_ln_relu_38