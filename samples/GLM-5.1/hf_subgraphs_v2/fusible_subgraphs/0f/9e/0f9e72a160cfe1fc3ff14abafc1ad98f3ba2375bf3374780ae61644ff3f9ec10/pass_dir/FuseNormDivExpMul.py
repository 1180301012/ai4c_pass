import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = torch.linalg.vector_norm(in_1, ord=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = torch.linalg.vector_norm(in_2, ord=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return tmp_6, tmp_4, tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_norm_div_exp_mul_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_2_ptr,
    out_4_ptr,
    out_6_ptr,
    n1,
    n2,
    BLOCK_SIZE: tl.constexpr,
):
    # Single program processes all elements to avoid atomics for small tensors
    pid = tl.program_id(0)
    if pid == 0:
        # Compute L2 norm of in_1 in float32 for numerical stability
        sum_sq_1 = 0.0
        for i in tl.range(0, n1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n1
            v1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_sq_1 += tl.sum(v1 * v1, 0)
        norm_1 = tl.sqrt(sum_sq_1)

        # Compute L2 norm of in_2 in float32
        sum_sq_2 = 0.0
        for i in tl.range(0, n2, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n2
            v2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_sq_2 += tl.sum(v2 * v2, 0)
        norm_2 = tl.sqrt(sum_sq_2)

        # Load exp(in_0) - scalar
        exp_val = tl.load(in_0_ptr).to(tl.float32)
        exp_val = tl.exp(exp_val)

        # Normalize in_1 and store to out_2
        for i in tl.range(0, n1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n1
            v1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            normalized_v1 = v1 / norm_1
            tl.store(out_2_ptr + offsets, normalized_v1, mask=mask)

        # Normalize in_2, compute exp(in_0) * normalized_in_2, store both
        for i in tl.range(0, n2, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n2
            v2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            normalized_v2 = v2 / norm_2
            scaled_v2 = exp_val * normalized_v2
            tl.store(out_4_ptr + offsets, normalized_v2, mask=mask)
            tl.store(out_6_ptr + offsets, scaled_v2, mask=mask)


@torch.fx.wrap
def fused_norm_div_exp_mul(in_0, in_1, in_2):
    n1 = in_1.numel()
    n2 = in_2.numel()
    BLOCK_SIZE = 256

    out_2 = torch.empty_like(in_1)
    out_4 = torch.empty_like(in_2)
    out_6 = torch.empty_like(in_2)

    fused_norm_div_exp_mul_kernel[(1,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_2_ptr=out_2,
        out_4_ptr=out_4,
        out_6_ptr=out_6,
        n1=n1,
        n2=n2,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_6, out_4, out_2


def replacement_func():
    return fused_norm_div_exp_mul