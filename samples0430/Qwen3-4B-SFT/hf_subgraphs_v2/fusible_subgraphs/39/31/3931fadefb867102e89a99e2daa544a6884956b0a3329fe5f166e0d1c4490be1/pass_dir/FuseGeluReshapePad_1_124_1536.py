import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N', 'TOTAL'],
)
@triton.jit
def gelu_reshape_pad_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,    # 310752
    TOTAL: tl.constexpr, # 311232  (1 * 249 * 768)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: GELU + reshape (free) + pad (zero-fill last row).
    Reads from the flat [1, 124, 1536] input and writes to flat [1, 249, 768] output.
    For output offsets >= N we write zeros (the padded row).
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    # padded rows >= TOTAL write 0.0
    is_padded = offsets >= N
    load_mask = mask & ~is_padded

    x = tl.load(in_ptr + offsets, mask=load_mask, other=0.0).to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu_out = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))

    # Write back with zeros for the one padded row
    out = tl.where(is_padded, 0.0, gelu_out)
    tl.store(out_ptr + offsets, out.to(in_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def gelu_reshape_pad_fused(in_0):
    """
    Fused: gelu + reshape(1,124,2,768) + reshape(1,248,768) + pad(0,0,0,1,'constant').
    in_0: [1, 124, 1536]
    returns: [1, 249, 768]
    """
    N = 1 * 124 * 1536    # 310752 — valid GELU+reshape output elements
    TOTAL = 1 * 249 * 768  # 311232 — final output elements (extra row is zeros)

    out = torch.empty((1, 249, 768), dtype=in_0.dtype, device=in_0.device)

    # grid is determined by autotuner-selected BLOCK_SIZE; pass TOTAL so
    # the autotune key includes N.
    def grid(meta):
        return ((TOTAL + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    gelu_reshape_pad_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        N=N,
        TOTAL=TOTAL,
    )

    return out


def replacement_func():
    return gelu_reshape_pad_fused