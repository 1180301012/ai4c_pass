import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_gelu_mul_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)

    # GELU (exact): x * 0.5 * (1.0 + erf(x / sqrt(2)))
    x_fp32 = x.to(tl.float32)
    gelu_x = x_fp32 * (0.5 + 0.5 * tl.math.erf(x_fp32 * 0.7071067811865476))

    # Element-wise multiply
    out = gelu_x.to(x.dtype) * y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_gelu_mul(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)

    # Adaptive block size: need enough blocks for parallelism on 56 SMs
    if N <= 65536:
        BLOCK_SIZE = 256
        num_warps = 2
    elif N <= 524288:
        BLOCK_SIZE = 1024
        num_warps = 4
    else:
        BLOCK_SIZE = 2048
        num_warps = 4

    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_gelu_mul_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )

    return out


def replacement_func():
    return fused_gelu_mul