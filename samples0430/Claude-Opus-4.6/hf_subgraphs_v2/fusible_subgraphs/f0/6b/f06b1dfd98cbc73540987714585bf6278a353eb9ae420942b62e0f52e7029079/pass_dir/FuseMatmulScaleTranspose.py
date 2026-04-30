import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_matmul_scale_kernel(
    in_2_ptr,
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    scale = tl.load(in_0_ptr).to(tl.float32)
    k_offsets = tl.arange(0, K)
    a = tl.load(in_2_ptr + row * K + k_offsets).to(tl.float32)
    b = tl.load(in_1_ptr + k_offsets).to(tl.float32)
    dot = tl.sum(a * b, axis=0)
    tl.store(out_ptr + row, dot * scale)


@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    out = torch.empty([in_2.shape[0], 1], dtype=in_2.dtype, device=in_2.device)
    fused_matmul_scale_kernel[(in_2.shape[0],)](
        in_2, in_1, in_0, out,
        K=in_2.shape[1],
        num_warps=1, num_stages=1,
    )
    return out


def replacement_func():
    return fused_matmul_scale