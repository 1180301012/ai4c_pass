import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input; cast to float32 for numerical stability
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    neg_x = -x_f32
    exp_neg = tl.exp(neg_x)
    sigmoid_val = 1.0 / (1.0 + exp_neg)
    out_f32 = x_f32 * sigmoid_val

    # Cast back to original dtype and store
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def silu_detatch_fwd(in_0, in_1, in_2):
    # SiLU on in_0 — the only real computation
    N = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    silu_kernel[grid](
        in_0,
        out,
        N,
    )
    # detach() is a no-op on stored tensors — the inputs pass through as-is
    return (in_1, in_2, out, out)


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return silu_detatch_fwd