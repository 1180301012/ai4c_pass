import torch
import triton
import triton.language as tl


# Pattern: contiguous + view(-1,64,64,384) + roll(shifts=(4,4),dims=(1,2)) + view(1,4096,384)
def pattern(in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 64, 64, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 4096, 384)
    return tmp_5


def replacement_args(in_3):
    return (in_3,)


@triton.jit
def _roll_noncontig_384(
    in3_ptr, out_ptr,
    H, W, C,
    H2, W2,
    s1, s2, s3, s4,
    shift,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    h = row_idx // W
    w = row_idx % W
    src_h = (h - shift + H) % H
    src_w = (w - shift + W) % W
    a = src_h // H2
    b = src_h % H2
    c = src_w // W2
    d = src_w % W2
    src_offset = a * s1 + b * s2 + c * s3 + d * s4

    offsets = tl.arange(0, BLOCK_C)
    mask = offsets < C
    x = tl.load(in3_ptr + src_offset + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row_idx * C + offsets, x, mask=mask)


@torch.fx.wrap
def triton_roll_64x64x384(in_3):
    """
    Fused: non-contiguous read of in_3 + roll → contiguous output [1,4096,384].
    Eliminates the separate contiguous() copy by reading in_3 directly via strides.
    Falls back to native PyTorch ops when strides are not suitable for Triton.
    """
    H, W, C = 64, 64, 384
    N = H * W   # 4096

    s = in_3.stride()
    s5 = int(s[5])

    if s5 == 1:
        H2 = int(in_3.shape[2])
        W2 = int(in_3.shape[4])
        s1, s2, s3, s4 = int(s[1]), int(s[2]), int(s[3]), int(s[4])
        rolled = torch.empty(N, C, dtype=in_3.dtype, device=in_3.device)
        _roll_noncontig_384[(N,)](
            in_3, rolled, H, W, C, H2, W2, s1, s2, s3, s4,
            shift=4, BLOCK_C=512, num_warps=4,
        )
        return rolled.view(1, N, C)
    else:
        tmp = in_3.contiguous().view(-1, H, W, C)
        return torch.roll(tmp, shifts=(4, 4), dims=(1, 2)).view(1, N, C)


def replacement_func():
    return triton_roll_64x64x384