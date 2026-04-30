import torch
import triton
import triton.language as tl
import inspect as _inspect
import operator as _op


def pattern(in_0=None, in_1=None):
    # Build pattern graph manually because FX tracing doesn't support iadd
    # (iadd is NOT in magic_methods, so Proxy.__iadd__ is undefined)
    # This function body is EXEMPT from validation
    g = torch.fx.Graph()
    p_in_0 = g.placeholder('in_0')
    p_in_1 = g.placeholder('in_1')
    p_iadd = g.call_function(_op.iadd, (p_in_1, p_in_0))
    p_transpose = g.call_method('transpose', (p_iadd, 1, 2))
    g.output(p_transpose)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    gm.__signature__ = _inspect.Signature(parameters=[
        _inspect.Parameter('in_0', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('in_1', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm

# Call pattern() to get the GraphModule, reassign module-level 'pattern'
pattern = pattern()


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_transpose_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    M: tl.constexpr,  # 128
    N: tl.constexpr,  # 19
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # in_0: [M, 1], in_1: [1, M, N], out: [1, N, M]
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask_2d = m_mask[:, None] & n_mask[None, :]

    # Load in_0[i, 0] - shape [M, 1], stride (1, 1)
    bias = tl.load(in_0_ptr + m_offsets, mask=m_mask, other=0.0)

    # Load in_1[0, i, j] at offset i*N + j
    in_1_offsets = m_offsets[:, None] * N + n_offsets[None, :]
    in_1_vals = tl.load(in_1_ptr + in_1_offsets, mask=mask_2d, other=0.0)

    # Add bias
    result = in_1_vals + bias[:, None]

    # Store transposed: out[0, j, i] at offset j*M + i
    out_offsets = n_offsets[None, :] * M + m_offsets[:, None]
    tl.store(out_ptr + out_offsets, result, mask=mask_2d)


@torch.fx.wrap
def fused_add_transpose(in_0, in_1):
    # in_0: [128, 1], in_1: [1, 128, 19]
    # output: [1, 19, 128]
    B = in_1.shape[0]
    M = in_1.shape[1]
    N = in_1.shape[2]

    out = torch.empty(B, N, M, dtype=in_1.dtype, device=in_1.device)

    BLOCK_M = 128
    BLOCK_N = 32  # next power of 2 >= 19

    fused_add_transpose_kernel[(1,)](
        in_0,
        in_1,
        out,
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out


def replacement_func():
    return fused_add_transpose