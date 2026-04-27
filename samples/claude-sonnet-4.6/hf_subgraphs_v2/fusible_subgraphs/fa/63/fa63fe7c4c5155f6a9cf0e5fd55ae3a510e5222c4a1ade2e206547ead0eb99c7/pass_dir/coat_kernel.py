import torch
import triton
import triton.language as tl
import operator


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=4),
    ],
    key=['D', 'HW'],
)
@triton.jit
def coat_tr_reshape_kernel(
    in2_ptr, out_ptr,
    stride_h, stride_n, stride_d,
    N, D, HW, Wout, total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    c = offs // HW
    sp = offs % HW
    h = c // D
    d = c % D
    n = sp + 1
    in_offs = h * stride_h + n * stride_n + d * stride_d
    v = tl.load(in2_ptr + in_offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v, mask=mask)


@torch.fx.wrap
def coat_full_replace(in_0, in_1, in_2, route):
    """
    route format: "D_Hout_Wout_a_b"
    Returns: (matmul, split0, split1, split2, tmp1)
    """
    parts = route.split('_')
    D    = int(parts[0])
    Hout = int(parts[1])
    Wout = int(parts[2])
    a    = int(parts[3])
    b    = int(parts[4])

    matmul = in_1 @ in_0

    C     = 8 * D
    N     = in_2.shape[2]
    HW    = Hout * Wout
    total = C * HW
    st    = in_2.stride()

    out = torch.empty((1, C, Hout, Wout), dtype=in_2.dtype, device=in_2.device)

    def grid(META):
        return (triton.cdiv(total, META['BLOCK']),)

    coat_tr_reshape_kernel[grid](
        in_2, out, st[1], st[2], st[3], N, D, HW, Wout, total,
    )

    tmp_1 = in_1[:, :, 1:, :]
    return (matmul, out[:, :a, :, :], out[:, a:a+b, :, :], out[:, a+b:, :, :], tmp_1)


def replacement_func():
    return coat_full_replace