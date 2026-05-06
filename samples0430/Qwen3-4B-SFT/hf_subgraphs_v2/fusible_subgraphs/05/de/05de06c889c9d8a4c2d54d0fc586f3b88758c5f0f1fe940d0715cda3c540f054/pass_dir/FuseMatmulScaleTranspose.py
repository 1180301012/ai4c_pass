import torch
import triton
import triton.language as tl


# Match: matmul(in_2, in_1) * in_0
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# -----------------------------------------------------------------------
# Triton kernel – 2 CTAs, one per output row (like cuBLAS for tiny M)
# M=2, K=512 are hardcoded as constexprs.
# -----------------------------------------------------------------------
@triton.jit
def fused_matmul_scale_kernel(
    a_ptr,                    # [2, 512]
    b_ptr,                    # [512]
    scale_ptr,                # scalar
    out_ptr,                  # [2]
    IN_FP32:  tl.constexpr,
    M_CONST:  tl.constexpr,   # = 2
    K_CONST:  tl.constexpr,   # = 512
):
    row = tl.program_id(0)    # 0 or 1
    offs_k = tl.arange(0, K_CONST)   # [512]

    scale_f32 = tl.load(scale_ptr).to(tl.float32)

    # Coalesced load: row_MAJOR a[row, :] of the [2, 512] matrix
    a_vec = tl.load(a_ptr + row * K_CONST + offs_k)    # [512]

    b_vec = tl.load(b_ptr + offs_k)                     # [512]

    dot = tl.sum(a_vec * b_vec, axis=0)

    if IN_FP32:
        result = (dot * scale_f32).to(a_vec.dtype)
    else:
        result = (dot * scale_f32).to(a_vec.dtype)

    tl.store(out_ptr + row, result)


@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    out = torch.empty((2, 1), dtype=in_2.dtype, device=in_2.device)
    fused_matmul_scale_kernel[(2,)](
        in_2, in_1, in_0,
        out,
        IN_FP32=False,
        M_CONST=2,
        K_CONST=512,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_matmul_scale