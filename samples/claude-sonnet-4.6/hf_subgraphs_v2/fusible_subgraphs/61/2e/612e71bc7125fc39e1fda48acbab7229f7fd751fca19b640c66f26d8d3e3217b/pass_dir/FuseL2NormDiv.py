import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: row-wise L2-norm then divide
# Matches:
#   tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
#   tmp_1 = in_1 / tmp_0
# ---------------------------------------------------------------------------
def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: fused per-row L2 normalisation for [N, D] bfloat16.
# Single pass: load → squared-sum reduction → rsqrt → multiply → store.
# One CTA per row; BLOCK_D = next_power_of_2(D) so the kernel is compiled
# once per unique D (768→1024, 1024→1024, 1152→2048).
# float32 accumulation for parity with PyTorch, bfloat16 output.
# ---------------------------------------------------------------------------
@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    D,
    stride_n,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(x_ptr + row * stride_n + cols, mask=mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(x * x, axis=0)
    inv_norm = tl.rsqrt(norm_sq)
    out = (x * inv_norm).to(tl.bfloat16)
    tl.store(out_ptr + row * stride_n + cols, out, mask=mask)


@torch.fx.wrap
def l2_normalize_triton(in_1):
    N, D = in_1.shape
    out = torch.empty_like(in_1)
    BLOCK_D = triton.next_power_of_2(D)
    l2_normalize_kernel[(N,)](
        in_1, out, D, D,
        BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out


def replacement_func():
    return l2_normalize_triton