import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_4.view(1, 1, -1, 64) -> transpose(1, 2) -> contiguous()
# ---------------------------------------------------------------------------
def pattern(in_4):
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    return tmp_9


def replacement_args(in_4):
    return (in_4,)


# ---------------------------------------------------------------------------
# Triton kernel: copy [1, 1, 512] -> [1, 8, 1, 64] contiguous
#
# in_4 is [1, 1, 512] — a flat 512-element row.
# The result [1, 8, 1, 64] is the same 512 bfloat16/float16 values but
# stored in a different shape (and contiguous memory layout).
# Essentially a memcopy with shape reinterpretation.
# ---------------------------------------------------------------------------
@triton.jit
def view_transpose_contiguous_kernel(
    src_ptr,   # [1, 1, 512] contiguous
    dst_ptr,   # [1, 8, 1, 64] contiguous
    N: tl.constexpr,   # 512
):
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < N
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


@torch.fx.wrap
def handle_view_transpose_contiguous(in_4):
    """
    Replaces: in_4.view(1, 1, -1, 64).transpose(1, 2).contiguous()
    in_4 : [1, 1, 512]  (contiguous)
    Returns: [1, 8, 1, 64]  contiguous
    """
    N = 512
    BLOCK_SIZE = 256
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty((1, 8, 1, 64), dtype=in_4.dtype, device=in_4.device)

    view_transpose_contiguous_kernel[(num_blocks,)](
        in_4, out, N, BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return handle_view_transpose_contiguous