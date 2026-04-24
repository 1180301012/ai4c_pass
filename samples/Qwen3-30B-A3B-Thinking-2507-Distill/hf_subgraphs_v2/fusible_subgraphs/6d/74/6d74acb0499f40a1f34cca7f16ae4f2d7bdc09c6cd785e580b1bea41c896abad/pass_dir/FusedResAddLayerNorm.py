import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def _fused_res_add_layernorm_kernel(
    in0_ptr,   # bias, shape [N]
    in1_ptr,   # weight, shape [N]
    in2_ptr,   # residual, shape [*, N]
    in3_ptr,   # hidden_states, shape [*, N]
    out_ptr,   # output, shape [*, N], dtype float32
    N,         # hidden dimension (e.g. 768)
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D grid: one block per row.
    num_warps=4 → 128 threads → 8 elements/thread → 128-bit vectorized loads for float32.
    ~10 blocks/SM on A30 → fewest waves for large M → best bandwidth utilization.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load and add inputs, promoting to float32
    in2 = tl.load(in2_ptr + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    in3 = tl.load(in3_ptr + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    x = in2 + in3

    # --- LayerNorm: mean ---
    mean = tl.sum(x, axis=0) / N

    # --- LayerNorm: variance ---
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    # --- Normalize ---
    rstd = 1.0 / tl.sqrt(var + 1e-7)
    x_norm = diff * rstd

    # --- Affine: weight * x_norm + bias ---
    w = tl.load(in1_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(in0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    tl.store(out_ptr + row * N + offsets, out, mask=mask, eviction_policy='evict_first')


@torch.fx.wrap
def fused_res_add_layernorm(in_0, in_1, in_2, in_3):
    """
    Fused: (in_3 + in_2).layer_norm(weight=in_1, bias=in_0)
    Returns float32 tensor, shape [..., N].
    """
    N = in_2.shape[-1]
    M = in_2.numel() // N

    out = torch.empty(in_2.shape, dtype=torch.float32, device=in_2.device)

    # num_warps=4 (128 threads): 8 elements/thread → 128-bit vectorized loads.
    # ~10 blocks/SM on A30 → fewest waves for large M → best bandwidth utilization.
    _fused_res_add_layernorm_kernel[(M,)](
        in_0, in_1, in_2, in_3, out,
        N=N,
        BLOCK_SIZE=1024,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_res_add_layernorm